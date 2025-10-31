# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Provides functionality for filling missing data on UGrid."""

import abc
import warnings

import numpy as np
from iris.cube import Cube
from pykdtree.kdtree import KDTree

from ugants.analysis.coord_transforms import convert_to_cartesian
from ugants.utils.cube import get_connectivity_indices


class FillABC(abc.ABC):
    """An abstract base class which provides an interface for fill operations.

    The abstract methods to be provided when subclassing the :class:`FillABC` are:

    * :meth:`calculate_fill_lookup` - Perform precalculation prior to filling
        missing data.
    * :meth:`__call__` - Perform the missing data fill on the supplied cube.

    Attributes
    ----------
    source : :class:`iris.cube.Cube`
        UGrid cube with missing data to be filled, identified by masked data.
    target_mask : :class:`iris.cube.Cube`
        Target mask to be applied to constrain the search, by default None.
        The filled cube will inherit this target mask.
    indices_to_be_filled : :class:`numpy.ndarray`
        A boolean array specifying the cells in the source data that need to be filled.
    indices_to_fill_from : :class:`numpy.ndarray`
        A boolean array specifying the cells in the source data that are
        considered valid fill candidates.

    """

    def __init__(self, source: Cube, target_mask: Cube = None):
        """Identify cells to fill, cells to fill from, and calculate_fill_lookup.

        Missing points are identified by either masked or nan values.
        The ``target_mask`` is used to constrain the valid fill points,
        so that only unmasked points are considered valid fill candidates.
        In addition, the ``target_mask`` is applied to the filled cube.

        The following truth table shows the behaviour of a given cell for the
        various combinations of being masked in ``source`` and ``target_mask``:

        ====================  ========================  =====================  ====================
        Masked in ``source``  Value in ``target_mask``  Cell requires filling  Valid fill candidate
        ====================  ========================  =====================  ====================
        False                 False                     False                  True
        False                 True                      False                  False
        True                  False                     True                   False
        True                  True                      False                  False
        ====================  ========================  =====================  ====================

        Parameters
        ----------
        source
            UGrid cube with missing data to be filled, identified by either NaN or masked data.
        target_mask
            Target mask to be applied to constrain the search, by default None.
            Cells that are ``True`` in the target mask are considered invalid fill candidates.
        """  # noqa: E501
        if (target_mask is not None) and (source.mesh != target_mask.mesh):
            raise ValueError("Source and target mask have different meshes")

        self.source = convert_nan_to_masked(source)

        self.target_mask = None
        if target_mask is not None:
            self._validate_target_mask(target_mask)
            self.target_mask = target_mask.copy(target_mask.data.astype(bool))

        self.indices_to_be_filled: np.ndarray
        self.identify_cells_to_fill()

        self.indices_to_fill_from: np.ndarray
        self.identify_valid_fill_cells()

        self.calculate_fill_lookup()

    @staticmethod
    def _validate_target_mask(target_mask):
        # Only integer or boolean dtypes allowed in target mask
        if (dtype := target_mask.data.dtype).kind not in ("i", "b"):
            raise TypeError(
                f"Unexpected target mask dtype: {dtype}, "
                "expected integer or boolean."
            )

        # No masked data allowed in target mask
        if np.ma.is_masked(target_mask.data):
            raise ValueError("Unexpected masked data present in target mask.")

        # Target mask should only contain values 0 or 1 (False or True).
        # Check the actual unique values set is a subset of the expected set {0, 1}.
        # This accounts for the case where the target mask is all 0, in which case
        # unique_values would be {0}.
        # Note this works even if the dtype is bool rather than int.
        if not ((unique_values := set(target_mask.data)) <= {0, 1}):
            raise ValueError(f"Unexpected values in target mask, got {unique_values}.")

    @abc.abstractmethod
    def calculate_fill_lookup(self):
        """Perform precalculation for filling missing data.

        Initialise any data structures used in the :meth:`__call__` method,
        and return ``None``.
        """
        pass

    @abc.abstractmethod
    def __call__(self, cube: Cube) -> Cube:
        """Perform the missing data fill on the supplied cube.

        Make use of any data structures created during the :meth:`calculate_fill_lookup`
        method.
        Return a new :class:`iris.cube.Cube` with the missing data filled.
        """
        pass

    def identify_cells_to_fill(self):
        """Identify cells within :attr:`source` cube to be filled.

        Sets the :attr:`indices_to_be_filled` attribute.

        Returns
        -------
        None
            In-place operation.
        """
        missing_in_source = np.ma.getmaskarray(self.source.data)
        if self.target_mask is not None:
            # find points which are masked in source and not 1 in target_mask
            not_target_mask = ~self.target_mask.data
            search_mask = missing_in_source & not_target_mask
        else:
            search_mask = missing_in_source

        if np.all(~search_mask):
            warnings.warn("No cells in the source cube require filling.", stacklevel=1)
        self.indices_to_be_filled = search_mask
        return None

    def identify_valid_fill_cells(self):
        """Identify cells within :attr:`source` cube which are valid fill candidates.

        Sets the :attr:`indices_to_fill_from` attribute.

        Returns
        -------
        None
            In-place operation.
        """
        valid_in_source = ~np.ma.getmaskarray(self.source.data)
        if self.target_mask is not None:
            valid_in_target = ~self.target_mask.data
            if np.all(valid_in_target):
                warnings.warn(
                    "All target mask data is unmasked, target mask has no effect.",
                    stacklevel=1,
                )
            valid_mask = valid_in_source & valid_in_target
        else:
            valid_mask = valid_in_source

        if np.all(~valid_mask):
            raise ValueError("No valid fill candidates in source cube.")
        self.indices_to_fill_from = valid_mask
        return None

    def __repr__(self) -> str:
        """Provide a string representation of the class instance."""
        classname = self.__class__.__name__
        msg = f"{classname}(source={self.source!r}, target_mask={self.target_mask!r})"
        return msg


class KDTreeFill(FillABC):
    """Fill points with data from nearest neighbour, identified using a KDTree.

    Warning
    -------
    The KDTree fill assumes that the data is located on a spherical geocentric
    coordinate system.
    Attempting to use this fill algorithm with a different coordinate system may lead
    to unexpected results.
    """

    def calculate_fill_lookup(self):
        """Train and query a KDTree to identify nearest neighbours.

        The location of every face of the source cube is converted to a 3D vector.
        This 'point cloud' is then split into two sets: 'points to be filled' and
        'points to fill from'. The 'points to fill from' are used to train a KDTree,
        which is then queried using the 'points to be filled'. This provides a mapping
        from every missing point in the source cube to the index of its nearest
        neighbour in the valid subset.
        """
        point_cloud = convert_to_cartesian(self.source)
        self.points_to_be_filled = point_cloud[self.indices_to_be_filled]
        self.points_to_fill_from = point_cloud[self.indices_to_fill_from]
        self.kdtree = KDTree(self.points_to_fill_from)
        _, self.nearest_neighbour_indices = self.kdtree.query(self.points_to_be_filled)
        return None

    def __call__(self, cube: Cube) -> Cube:
        """
        Fill missing points in ``cube``.

        Missing data are filled with nearest neighbour data using the precalculated
        KDTree to look up valid data.

        Parameters
        ----------
        cube
            Cube with missing points to be filled.

        Returns
        -------
        :class:`iris.cube.Cube`
            Cube with missing points filled with data from nearest neighbour.
        """
        if cube.mesh != self.source.mesh:
            raise ValueError("Provided cube and source cube have different meshes")

        target = convert_nan_to_masked(cube)

        # NOTE: nearest_neighbour_indices references the points_to_fill_from array,
        # NOT the data array of the full cube
        valid_fill_data = target.data[self.indices_to_fill_from]
        target.data[self.indices_to_be_filled] = valid_fill_data[
            self.nearest_neighbour_indices
        ]

        if self.target_mask is not None:
            target.data.mask = self.target_mask.data
        return target


def flood_fill(cube: Cube, seed_point: int, fill_value: float):
    """Fill a contiguous region in cube, starting at seed point.

    The flood fill algorithm identifies a contiguous region, starting
    at the ``seed_point`` and extending outwards to neighbouring cells
    with the same data value as the seed point. Neighbours are identified
    using the mesh's :obj:`~iris.experimental.ugrid.mesh.Mesh.face_face_connectivity`.

    Parameters
    ----------
    cube :
        Cube to flood fill.
    seed_point :
        Index of the cube's data array to start the flood fill.
    fill_value :
        Value with which to fill all cells in the contiguous region.
        To mask out the region, set ``fill_value=np.ma.masked``.

    Returns
    -------
    :class:`iris.cube.Cube`
        Flood filled copy of the original cube.

    Raises
    ------
    ValueError
        If the provided cube has a non-horizontal dimension.
    ValueError
        If the specified seed point already has the specified fill value.

    Note
    ----
    There is no way to specify an extended neighbourhood in the same way as regular
    ANTS. Only immediate neighbours as identified by the cube's
    :obj:`~iris.experimental.ugrid.mesh.Mesh.face_face_connectivity`
    are considered, not diagonals.

    Wraparound is always enabled since there is no
    meaningful boundary in the unstructured grid.

    Currently the flood fill can only be used on horizontal data. Cubes with
    vertical levels, time coordinates or other non-mesh coordinates are not supported.
    """
    cube_dimensions_number = len(cube.shape)
    if cube_dimensions_number != 1:
        raise ValueError(
            f"The provided cube has {cube_dimensions_number} data dimensions, "
            "expected only 1 horizontal dimension."
        )
    filled_cube = cube.copy()

    # connectivity_indices is a numpy array specifying the nodes connected to each face.
    # shape = (n_faces, 4) since each face is quadrilateral.
    connectivity_indices = get_connectivity_indices(cube, "face_face_connectivity")

    seed_value = cube.data[seed_point]

    if seed_value == fill_value:
        raise ValueError(
            f"The value at location {seed_point} already has the "
            f"fill value {fill_value}."
        )

    ## MASKS
    # Starting setup: contiguous region = boundary = seed cell
    contiguous_region = np.zeros(cube.shape, dtype=bool)
    contiguous_region[seed_point] = True
    boundary = contiguous_region.copy()

    # All cells that contain valid data
    equal_to_seed_value = cube.data == seed_value

    while boundary.any():
        # Mask indentifying all cells that border the current boundary
        adjacent_to_boundary = np.zeros(cube.shape, dtype=bool)
        indices_adjacent_to_boundary = connectivity_indices[boundary].flatten()
        adjacent_to_boundary[indices_adjacent_to_boundary] = True

        # Next layer of cells to add to the contiguous region satisfies three criteria
        # 1. Adjacent to the current boundary of the contriguous region
        # 2. Data value equal to that of the seed cell
        # 3. Not already part of the contiguous region
        # The reason for this last criterion is so that we can use the next_layer
        # from this iteration as the boundary for the next iteration
        next_layer = adjacent_to_boundary & equal_to_seed_value & ~contiguous_region
        contiguous_region = contiguous_region | next_layer
        boundary = next_layer

    filled_cube.data[contiguous_region] = fill_value
    return filled_cube


def convert_nan_to_masked(cube: Cube) -> Cube:
    """Mask NaN values in the provided ``cube``.

    Parameters
    ----------
    cube
        Source cube with missing data identified by ``np.nan`` values.

    Returns
    -------
    :class:`iris.cube.Cube`
        Cube with NaN values masked.
    """
    target = cube.copy()
    target.data = np.ma.masked_where(np.isnan(target.data), target.data)
    return target
