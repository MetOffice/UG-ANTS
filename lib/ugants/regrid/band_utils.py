#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Utility functions for band regridding."""

import math
from itertools import pairwise

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import Connectivity, Mesh


def mesh_to_cube(mesh, dtype: np.dtype = np.float64):
    """Turn a mesh into a :class:`iris.cube.Cube` with data using ``np.nan``.

    Parameters
    ----------
    mesh : :class:`iris.experimental.ugrid.mesh.Mesh`
        The mesh to be converted to an :class:`iris.cube.Cube`
    dtype : :class:`numpy.dtype`
        The data type used for the cube.data :class:`numpy.ndarray`, by default
        ``np.float64``.

    Returns
    -------
    :class:`iris.cube.Cube`
        The providied mesh as an :class:`iris.cube.Cube`.
    """
    data = np.full(mesh.face_coords.face_x.shape[0], np.nan, dtype=dtype)
    cube = Cube(data)

    mesh_coordinates = mesh.to_MeshCoords("face")
    for coord in mesh_coordinates:
        cube.add_aux_coord(coord, 0)

    return cube


def generate_band_bounds(start: float, stop: float, n_bands: int):
    """Generate a list of tuples representing bands of latitude or longitude.

    Divides an arbitrary interval, defined by ``start`` and ``stop``, into equally
    spaced subdivisions.

    Example
    -------
    >>> generate_band_bounds(0.0, 15.0, 3)
    [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]

    Parameters
    ----------
    start : float
        The start latitude/longitude limit.
    stop : float
        The end latitude/longitude limit.
    n_bands : int
        Number of bands to generate.

    Returns
    -------
    list[tuple[float, float]]
        A list of tuples, with each tuple defining a band.

    Raises
    ------
    ValueError
        When `n_bands` is less than 2.
    """
    if n_bands < 2:
        raise ValueError(f"n_bands must be 2 or more. Only got {n_bands}")

    bands_spacing = np.linspace(start, stop, n_bands + 1)
    bands = list(pairwise(bands_spacing))

    return bands


def find_cell_centres_within_latitude_bounds(
    cube, bounds: tuple, min_inclusive: bool = True, max_inclusive: bool = True
):
    """Return a boolean mask of cells where the centres are within latitude bounds.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        An :class:`iris.cube.Cube` with a mesh.
    bounds : tuple[float, float]
        The latitude bounds to check.
    min_inclusive : bool
        Include cell centres equal to the minimum latitude, by default True
    max_inclusive : bool
        Include cell centres equal to the maximum latitude, by default True

    Returns
    -------
    :class:`numpy.ndarray`
        An array of bools representing the face indices to subset.
    """
    min_bound, max_bound = min(bounds), max(bounds)
    points = cube.mesh.face_coords.face_y.points

    index_min = min_bound <= points if min_inclusive else min_bound < points
    index_max = points <= max_bound if max_inclusive else points < max_bound

    indices_mask = np.logical_and(index_min, index_max)

    return indices_mask


def subset_mesh_cube_by_indices(cube: Cube, indices):
    """Create a subset of a cube with a mesh by indexing its mesh dimension.

    Parameters
    ----------
    cube :  :class:`iris.cube.Cube`
        A cube containing the mesh to be subset.
    indices :  :class:`numpy.ndarray`
        A boolean or integer array representing the indices of the mesh dimension
        to be sliced.

    Returns
    -------
    :class:`iris.cube.Cube`
        The subsetted cube.

    Note
    ----
    Mesh cube subsetting is currently only available for data located on faces.
    """
    if cube.mesh is None:
        raise ValueError(
            "The provided cube does not have a mesh, expected one to be present."
        )
    if cube.location != "face":
        raise ValueError(
            "Mesh subsetting is currently only available for data located "
            f"on faces, the provided cube has location {cube.location}."
        )
    # index on the mesh dimension of the cube only, do not slice other dimensions
    slices = tuple(
        indices if dim == cube.mesh_dim() else slice(None) for dim in range(cube.ndim)
    )
    subset_cube = cube[slices]
    subset_cube = reconstruct_mesh_cube(subset_cube, cube.mesh_dim())

    # Copy specific metadata on the subset mesh from the old mesh.  Note that we
    # can't do this via a better method as there's no public method or
    # attribute to get the relevant fields from the MeshMetadata object.
    subset_cube.mesh.var_name = cube.mesh.var_name
    subset_cube.mesh.standard_name = cube.mesh.standard_name
    subset_cube.mesh.long_name = cube.mesh.long_name
    subset_cube.mesh.units = cube.mesh.units
    subset_cube.mesh.node_dimension = cube.mesh.node_dimension
    subset_cube.mesh.edge_dimension = cube.mesh.edge_dimension
    subset_cube.mesh.face_dimension = cube.mesh.face_dimension

    return subset_cube


def reconstruct_mesh_cube(cube: Cube, mesh_dim: int):
    """Reconstruct a cube with a mesh that has been broken into AuxCoords.

    When a cube is subset along a mesh dimension, the MeshCoords will be converted
    to AuxCoords, so the cube will no longer have a mesh. This function reconstructs
    the mesh from the AuxCoords, and adds corresponding MeshCoords to the cube.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube with a broken mesh to be reconstructed.
    mesh_dim : int
        The dimension of the cube which the mesh describes.

    Returns
    -------
    :class:`iris.cube.Cube`
        A copy of the original cube with a reconstructed mesh.

    Notes
    -----
    Note
    ----
    The reconstructed cube's mesh :obj:`~iris.cube.Cube.location` is set to "face".
    Other mesh locations (edge, node) are not supported.

    Note
    ----
    The reconstructed mesh's :obj:`~iris.experimental.ugrid.Mesh.face_node_connectivity`
    will be 0 indexed.

    Note
    ----
    Face order is preserved, but node order is not preserved under reconstruction.

    """
    if cube.mesh:
        raise ValueError("The provided cube already has a mesh.")

    reconstructed_cube = cube.copy()
    mesh = Mesh.from_coords(*reconstructed_cube.coords(dimensions=mesh_dim))
    # This mesh is not complete, nodes are duplicated
    #
    # Example
    # -------
    # This example 2-face mesh will be used to illustrate the numpy array operations
    #
    # Initial mesh:
    #  10 [4]------[5,6]------[7]
    #      |         |         |
    # y    |   (0)   |   (1)   |
    #      |         |         |
    #   0 [0]------[1,2]------[3]
    #      0        10         20
    #                x
    # Nodes 1 and 2 are coincident at (10, 0)
    # Nodes 5 and 6 are coincident at (10, 10)
    #
    # Nodes 1 and 5 are connected to face 0, nodes 2 and 6 are connected to face 1
    #
    # In terms of arrays, this looks like:
    # node_x = [0, 10, 10, 20,  0, 10, 10, 20]
    # node_y = [0,  0,  0,  0, 10, 10, 10, 10]
    # face_node_indices = [[0, 1, 5, 4], [2, 3, 7, 6]]
    #
    # Target:
    #  10 [1]-------[3]-------[5]
    #      |         |         |
    # y    |   (0)   |   (1)   |
    #      |         |         |
    #   0 [0]-------[2]-------[4]
    #      0        10         20
    #                x
    # In terms of arrays, this looks like:
    # node_x = [0,  0, 10, 10, 20, 20]
    # node_y = [0, 10,  0, 10,  0, 10]
    # face_node_indices = [[0, 2, 3, 1], [2, 4, 5, 3]]

    # The initial mesh nodes:
    # nodes_xy = [[0, 10, 10, 20,  0, 10, 10, 20],
    #             [0,  0,  0,  0, 10, 10, 10, 10]]
    nodes_xy = np.stack(
        [mesh.node_coords.node_x.points, mesh.node_coords.node_y.points]
    )

    # Reduce nodes_xy to an array of unique (x,y) pairs:
    # unique_nodes_xy = [[0,  0, 10, 10, 20, 20],
    #                    [0, 10,  0, 10,  0, 10]]
    #
    # Also return the indices at which each (x,y) pair in nodes_xy can be found in
    # the reduced array unique_nodes_xy:
    # first_instances = [0, 2, 2, 4, 1, 3, 3, 5]
    #
    # The first_instances array maps nodes from the initial mesh to nodes in the
    # deduplicated mesh:
    # * node 0 in original mesh maps to node 0 in deduplicated mesh
    # * nodes 1 and 2 in original mesh both map to node 2 in deduplicated mesh
    # * node 3 in original mesh maps to node 4 in deduplicated mesh
    # * node 4 in original mesh maps to node 1 in deduplicated mesh
    # * nodes 5 and 6 in original mesh both map to node 3 in deduplicated mesh
    # * node 7 in original mesh maps to node 5 in deduplicated mesh
    unique_nodes_xy, first_instances = np.unique(
        nodes_xy,
        return_inverse=True,
        axis=1,
    )
    new_node_x = AuxCoord(
        unique_nodes_xy[0], **mesh.node_coords.node_x.metadata._asdict()
    )
    new_node_y = AuxCoord(
        unique_nodes_xy[1], **mesh.node_coords.node_y.metadata._asdict()
    )

    # Remap face node connectivity using first_instances,
    # since the nodes have been reordered by deduplication
    original_face_node_connectivity = mesh.face_node_connectivity
    original_face_node_indices = (
        original_face_node_connectivity.indices_by_location()
        - original_face_node_connectivity.start_index
    )
    # Re-index the face node connectivity:
    # original_face_node_indices = [[0, 1, 5, 4], [2, 3, 7, 6]]
    # first_instances = [0, 2, 2, 4, 1, 3, 3, 5]
    # new_face_node_indices = [[0, 2, 3, 1], [2, 4, 5, 3]]
    new_face_node_indices = first_instances[original_face_node_indices]
    new_face_node_indices = np.ma.masked_array(
        new_face_node_indices,
        mask=original_face_node_connectivity.indices_by_location().mask,
    )
    # New face node connectivity will have the same metadata as the original,
    # but with start index of 0
    new_connectivity_kwargs = original_face_node_connectivity.metadata._asdict()
    new_connectivity_kwargs["start_index"] = 0
    new_face_node_connectivity = Connectivity(
        new_face_node_indices, **new_connectivity_kwargs
    )

    # Replace mesh's node coords and face node connectivity with deduplicated versions
    mesh.add_coords(node_x=new_node_x, node_y=new_node_y)
    mesh.add_connectivities(new_face_node_connectivity)

    # Convert new mesh to MeshCoords and add them to the reconstructed cube,
    # replacing the placeholder AuxCoords
    mesh_coords = mesh.to_MeshCoords(location="face")
    for coord in mesh_coords:
        reconstructed_cube.remove_coord(coord.name())
        reconstructed_cube.add_aux_coord(coord, mesh_dim)

    return reconstructed_cube


def cube_subset_latitude_bounds(mesh, coord, fraction=0.05):
    """Calculate latitude bounds that subset the cube so that it encloses the mesh.

    Parameters
    ----------
    mesh :  :class:`iris.experimental.ugrid.Mesh`
        The target mesh.
    coord :  :class:`iris.coords.Coord`
        The latitude coord.

    Returns
    -------
    tuple
        Latitude bounds required to subset a cube.
    """
    coord = coord.copy()

    if not coord.has_bounds():
        coord.guess_bounds()

    mesh_min_lat = min(mesh.node_coords.node_y.points)
    mesh_max_lat = max(mesh.node_coords.node_y.points)

    coord_min_lat, coord_max_lat = coord.collapsed().bounds[0]

    if mesh_min_lat < coord_min_lat or mesh_max_lat > coord_max_lat:
        raise ValueError(
            f"Coord bounds {coord_min_lat, coord_max_lat} do not enclose mesh"
            f" bounds {mesh_min_lat, mesh_max_lat}."
        )

    index_of_min_lat = coord.nearest_neighbour_index(mesh_min_lat)
    index_of_max_lat = coord.nearest_neighbour_index(mesh_max_lat)

    monotonic, direction = iris.util.monotonic(coord.points, return_direction=True)

    n_rows = abs(index_of_min_lat - index_of_max_lat) + 1
    padding = int(math.ceil(n_rows * fraction))

    if monotonic and direction == 1.0:
        index_of_padded_min_lat = max(index_of_min_lat - padding, 0)
        index_of_padded_max_lat = min(index_of_max_lat + padding, len(coord.points) - 1)
    elif monotonic and direction == -1.0:
        index_of_padded_max_lat = max(index_of_max_lat - padding, 0)
        index_of_padded_min_lat = min(index_of_min_lat + padding, len(coord.points) - 1)
    else:
        raise ValueError("The provided coord is not monotonic.")

    latitude_lower = min(coord.bounds[index_of_padded_min_lat])
    latitude_upper = max(coord.bounds[index_of_padded_max_lat])

    return latitude_lower, latitude_upper


def constrain_source_cube_latitude(cube, bounds, padding=0.0):
    """Constrain a cube based on latitude bounds and optional padding.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        Source cube to constrain.
    bounds : tuple[float, float]
        Lower and upper latitude limit.
    padding : float
        Optional padding to be applied to both limits.

    Returns
    -------
    :class:`iris.cube.Cube`
    """
    minimum_latitude = max(min(bounds) - padding, -90.0)
    maximum_latitude = min(max(bounds) + padding, +90.0)

    latitude_extent = iris.coords.CoordExtent(
        "latitude",
        minimum_latitude,
        maximum_latitude,
    )
    constrained_cube = cube.intersection(latitude_extent)

    return constrained_cube


def split_cube(cube, n_slices, axis="Y"):
    """Split a cube along the given dimension into approximately equal sub-cubes.

    The generated sub-cubes will be non-overlapping.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube to split.
    n_slices : int
        Total number of sub-cubes to return.
    axis : str
        Axis to split over.  Valid values are "Y" (default) to split on
        latitude, or "X" to split on longitude.  This is case insensitive.

    Returns
    -------
    :class:`iris.cube.CubeList`
        The sub cubes as a cube list.

    Raises
    ------
    ValueError
        If the axis not one of "X" or "Y".
    """
    if axis.upper() not in ["X", "Y"]:
        raise ValueError(f'Axis must be one of "X" or "Y".  Got: "{axis}".')

    shape = len(cube.shape)
    if shape != 2:
        raise ValueError(f"Input cube must be 2 dimensional.  Got {shape} dimensions.")

    split_coord = cube.coord(axis=axis, dim_coords=True)
    dim_length = split_coord.shape[0]

    indices = np.linspace(0, dim_length, n_slices + 1, dtype=np.int64)
    sub_cubes = CubeList()

    for start, stop in pairwise(indices):
        slice_list = [
            slice(start, stop) if coord == split_coord else Ellipsis
            for coord in cube.dim_coords
        ]
        sub_cubes.append(cube[tuple(slice_list)])

    return sub_cubes


def _add_padding(minimum, maximum, padding_fraction=0.1):
    """Pad minimum and maximum by the padding.

    Assumes minimum and maximum are latitude bounds - the approach used for
    padding may or may not be suitable for other cass.

    Approach is to scale minimum and maximum by padding_fraction x delta
    between minimum and maximum.

    Parameters
    ----------
    minimum : int
        Lowest bound.
    maximum : int
        Highest bound.
    padding_fraction : float
        Fraction by which to pad the minimum and maximum values.  Defaults to
        0.1.

    """
    if minimum >= maximum:
        raise ValueError(
            f"Minimum bound {minimum} is not below maximum bound {maximum}."
        )
    delta = maximum - minimum
    adjustment = padding_fraction * delta
    minimum = minimum - adjustment
    maximum = maximum + adjustment
    return minimum, maximum


def get_faces_that_overlap_bounds(cube, bounds, index=1):
    """Get the indices of faces with one or more nodes within given latitude bounds.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        A cube with associated mesh.
    bounds : tuple[float, float]
        The latitude bounds to check for overlapping faces.

    Returns
    -------
    :class:`numpy.ndarray`

    """
    min_lat, max_lat = min(bounds), max(bounds)
    min_lat, max_lat = _add_padding(min_lat, max_lat)
    node_indices = np.ravel(cube.mesh.face_node_connectivity.indices)
    node_latitudes = cube.mesh.node_coords.node_y.points[node_indices - index]
    node_mask = (min_lat <= node_latitudes) & (node_latitudes <= max_lat)
    face_node_mask = np.reshape(
        node_mask, cube.mesh.face_node_connectivity.indices.shape
    )

    return np.any(face_node_mask, axis=1)


def cube_latitude_bounds(cube):
    """Get the max and min of the latitude bounds from a cube.

    A copy of the "latitude" coordinate is taken in order to avoid mutating
    the provided cube if guessing bounds.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube to find latitude limits of.

    Returns
    -------
    tuple[float, float]
        The minimum and maximum latitude of the cube.
    """
    coord = cube.coord("latitude").copy()
    if not coord.has_bounds():
        coord.guess_bounds()

    return coord.bounds.min(), coord.bounds.max()
