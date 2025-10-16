# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Helper functions for unit tests.

Should only be used in unittests.  If any of this functionality is needed in
library or application code, it should be moved out of here and into the
library.  Any tests should then be updated to use the library versions.

"""

from itertools import pairwise

import geovista as gv
import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.experimental.ugrid import Connectivity, Mesh
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube


def mesh_cube(number_of_levels=0, include_mesh_dimcoord=False, **sample_mesh_kwargs):
    """Generate an iris cube containing a mesh.

    The cube may contain a structured "levels" dimension and an unstructured mesh
    mesh dimension. The data of the returned cube will be an array of zeros.

    Parameters
    ----------
    number_of_levels : :obj:`int`, optional
        Number of vertical levels, by default 0.
        If zero, then no levels dimension is added to the cube,
        so it will contain only an unstructured mesh dimension.
    include_mesh_dimcoord: bool, optional
        If True, include a DimCoord on the mesh dimension called "i_mesh_face",
        representing the indices of the faces.
    **sample_mesh_kwargs :
        Keyword arguments to pass to :func:`iris.tests.stock.mesh.sample_mesh`.

    Returns
    -------
    :class:`iris.cube.Cube`
        A test iris cube containing a mesh.

    """
    # Work around bug in iris sample_mesh_cube function: providing n_z=0 creates a
    # vertical dimension coordinate of length zero, rather than a scalar coordinate.
    # The work around to produce a cube with no vertical levels is to add one vertical
    # level, then remove it.
    n_z = 1 if number_of_levels == 0 else number_of_levels
    mesh = sample_mesh(**sample_mesh_kwargs)
    cube = sample_mesh_cube(mesh=mesh, n_z=n_z)
    if number_of_levels == 0:
        cube.remove_coord("level")
        cube = cube[0]
    if not include_mesh_dimcoord:
        cube.remove_coord("i_mesh_face")
    return cube


def regular_lat_lon_mesh(
    min_lon: float = 0.0,
    max_lon: float = +180.0,
    min_lat: float = 0.0,
    max_lat: float = +90.0,
    shape: tuple = (20, 30),
):
    """Generate a mesh resembling a regular lat/lon grid.

    Both node and face indexing iterate over latitude first, longitude second
    (see the example below for an illustration).

    Example
    -------
    >>> regular_lat_lon_mesh(0, 15, 0, 10, (3, 2))

    The following diagram illustrates the indexing of the
    nodes [i] and the faces (j) of the dummy mesh.

    10.0  [2]-------[5]-------[8]-------[11]
           |         |         |         |
    7.5    |   (1)   |   (3)   |   (5)   |
           |         |         |         |
    5.0   [1]-------[4]-------[7]-------[10]
           |         |         |         |
    2.5    |   (0)   |   (2)   |   (4)   |
           |         |         |         |
    0.0   [0]-------[3]-------[6]-------[9]

          0.0  2.5  5.0  7.5 10.0 12.5 15.0

    Parameters
    ----------
    min_lon : float
        Start longitude, by default 0.0
    max_lon : float
        End longitude, by default +180.0
    min_lat : float
        Start latitude, by default 0.0
    max_lat : float
        End latitude, by default +90.0
    shape : tuple
        Resolution of the mesh (lonxlat), by default (20, 30)
    """
    # Create node coordinates
    lons = np.linspace(min_lon, max_lon, shape[0] + 1)
    lats = np.linspace(min_lat, max_lat, shape[1] + 1)
    ys, xs = np.meshgrid(lats, lons)
    node_lat = AuxCoord(ys.ravel(), standard_name="latitude")
    node_lon = AuxCoord(xs.ravel(), standard_name="longitude")
    node_coords = [(node_lat, "y"), (node_lon, "x")]

    # Create face centre coordinates
    lon_centres_1d = [(x + y) / 2 for x, y in pairwise(lons)]
    lat_centres_1d = [(x + y) / 2 for x, y in pairwise(lats)]
    lat_centres_2d, lon_centres_2d = np.meshgrid(lat_centres_1d, lon_centres_1d)
    face_lat = AuxCoord(lat_centres_2d.ravel(), standard_name="latitude")
    face_lon = AuxCoord(lon_centres_2d.ravel(), standard_name="longitude")
    face_coords = [(face_lat, "y"), (face_lon, "x")]

    # Create face_node_connectivity
    indices = gv.Transform._create_connectivity_m1n1(xs.shape)
    indices = np.ma.masked_array(indices)
    connectivity = Connectivity(indices, cf_role="face_node_connectivity")

    mesh = Mesh(
        topology_dimension=2,
        node_coords_and_axes=node_coords,
        face_coords_and_axes=face_coords,
        connectivities=connectivity,
    )

    return mesh


def regular_grid_global_cube(n_lat: int, n_lon: int):
    """Generate a regular cube containing synthetic data on a global lat-lon grid.

    The cube will have two dimensions:

    - Dimension 0: latitude, n_lat points, generated by :func:`latitude_coord_regular`
    - Dimension 1: longitude, n_lon points,
      generated by :func:`circular_longitude_coord_regular`

    The cube's synthetic data is generated with :func:`numpy.arange`.

    Parameters
    ----------
    n_lat: int
        Number of latitude points
    n_lon: int
        Number of longitude points

    Returns
    -------
    :class:`iris.cube.Cube`
        A global Cube containing synthetic data

    """
    data = np.ma.arange(n_lat * n_lon).reshape((n_lat, n_lon))
    dim_coords_and_dims = [
        (regular_grid_latitude_coord(n_lat), 0),
        (regular_grid_circular_longitude_coord(n_lon), 1),
    ]
    cube = Cube(
        data=data, dim_coords_and_dims=dim_coords_and_dims, long_name="sample_data"
    )
    return cube


def regular_grid_circular_longitude_coord(n_lon: int):
    """Generate a circular longitude dimension coordinate.

    The coordinate will contain both points and bounds. The bounds are equally spaced
    and cover the range -180.0 to +180.0. The points are the midpoints of the bounds.

    Parameters
    ----------
    n_lon : int
        Number of longitude points

    Returns
    -------
    :class:`iris.coords.DimCoord`
        A DimCoord of longitude points and bounds
    """
    longitude_bound_values = np.linspace(-180, 180, n_lon + 1)
    longitude_bounds = np.array(
        [[lower, upper] for lower, upper in pairwise(longitude_bound_values)]
    )
    longitude_points = longitude_bounds.mean(axis=1)
    longitude_coord = DimCoord(
        points=longitude_points,
        standard_name="longitude",
        bounds=longitude_bounds,
        circular=True,
        units="degrees",
    )
    return longitude_coord


def regular_grid_latitude_coord(n_lat: int):
    """Generate a  latitude dimension coordinate.

    The coordinate will contain both points and bounds. The bounds are equally spaced
    and cover the range -90.0 to +90.0. The points are the midpoints of the bounds.

    Parameters
    ----------
    n_lat : int
        Number of latitude points

    Returns
    -------
    :class:`iris.coords.DimCoord`
        A DimCoord of latitude points and bounds
    """
    latitude_bound_values = np.linspace(-90, 90, n_lat + 1)
    latitude_bounds = np.array(
        [[lower, upper] for lower, upper in pairwise(latitude_bound_values)]
    )
    latitude_points = latitude_bounds.mean(axis=1)
    latitude_coord = DimCoord(
        points=latitude_points,
        standard_name="latitude",
        bounds=latitude_bounds,
        units="degrees",
    )
    return latitude_coord


def four_vertical_levels_cube():
    """Create a cube with a structured vertical dimension and a horizontal mesh.

    The returned cube has 4 vertical levels.

    The vertical dimension has three coordinates:

    - model_level_number (DimCoord)
    - level_height (AuxCoord)
    - sigma (AuxCoord)

    The returned cube has no zeroth level.
    The model_level_number coordinate starts at 1.

    The points of the coordinates represent theta levels,
    the bounds represent rho levels.

    Sigma values for level 'i' below the first constant rho level are calculated
    according to the formula:

    sigma[i] = (1 - (level_height[i] / level_height[j])) ** 2

    where 'j' is the "first constant rho level", i.e. the first rho level at which
    sigma is zero. Note that 'i' can represent a point (theta) or a bound (rho).
    See Appendix A of the F03 specification (sigma is called 'C' in this paper):
    https://code.metoffice.gov.uk/doc/um/latest/papers/umdp_F03.pdf

    For the purposes of this test function, these sigma values are hard-coded and
    rounded to 3 decimal places.
    For all levels above the first constant rho level, sigma is 0.

    ---- rho (bound)
    ==== theta (point)

    model            level
    level            height  sigma
    number
       --------------- 20.0   0.0

    4  =============== 16.0   0.0

       --------------- 12.0   0.0  > first constant rho level

    3  ===============  8.0   0.111

       ---------------  6.0   0.25

    2  ===============  4.0   0.444

       ---------------  3.0   0.563

    1  ===============  2.0   0.694

       ---------------  0.0   1.0
    """
    cube = mesh_cube(number_of_levels=4)
    cube.rename("vertical_cube")

    vertical_attributes = {"positive": "up"}

    model_level_number_coord = DimCoord(
        [1, 2, 3, 4], "model_level_number", units="1", attributes=vertical_attributes
    )
    cube.remove_coord(cube.coord("level"))
    cube.add_dim_coord(model_level_number_coord, 0)

    level_height_coord = AuxCoord(
        points=[2.0, 4.0, 8.0, 16.0],
        bounds=[[0.0, 3.0], [3.0, 6.0], [6.0, 12.0], [12.0, 20.0]],
        var_name="level_height",
        units="m",
        attributes=vertical_attributes,
    )
    cube.add_aux_coord(level_height_coord, 0)

    sigma_coord = AuxCoord(
        points=[0.694, 0.444, 0.111, 0.0],
        bounds=[[1.0, 0.563], [0.563, 0.25], [0.25, 0.0], [0.0, 0.0]],
        var_name="sigma",
        units="1",
    )
    cube.add_aux_coord(sigma_coord, 0)

    return cube
