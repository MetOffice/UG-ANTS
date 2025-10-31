# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.

# See LICENSE.txt in the root of the repository for full licensing details.
"""
Demonstration examples of filter classes.

All built on :class:`ugants.filter.generic.UnstructuredFilterABC`.
"""

import numpy as np
from iris.cube import Cube

from ugants.utils import move_one_dimension

from .generic import UnstructuredFilterABC as _MeshFilter


class NullMeshFilter(_MeshFilter):
    """
    An example, minimal  spatial filter.

    Derived from :class:`~ugants.filter.generic.UnstructuredFilterABC`.
    This filter makes no change to the input at all.
    """

    def __init__(self, cube: Cube):
        super().__init__(cube)

    def inner_filter(self, cube, original_standard_name):
        """Inner filter routine : return input unchanged."""
        return cube


class FaceNeighbourhoodFilter(_MeshFilter):
    """
    A simple spatial filter based on the unstructured connectivity.

    Makes a weighted combination of each face value with the average of its immediate
    neighbours.  This enables a general smoothing or sharpening effect similar to a
    3x3 window filter.

    Notes
    -----
    * only applicable to face-based data
    * the mesh must have faces, and a face-face connectivity.
    * the calculation makes no allowance for different effective distances or areas of
      the various neighbouring faces.  Also, different mesh faces may have different
      *numbers* of neighbours.  This means that the results do not have any definite
      physical meaning, and at best only approximate a true spatial filter.
    * the calculation does not correctly account for masked input data.  This is
      because, for simplicity and efficiency, it averages over a fixed set of
      neighbours at each point, without excluding masked datapoints.

      So, this operation roughly emulates a traditional 'patch kernel' operation, but
      in the absence of a regular grid the results are less reliable.

    """

    def __init__(
        self,
        cube: Cube,
        *,
        central_fraction: float = 1.0,
        neighbours_fraction: float = 1.0,
    ):
        """
        Create a face-neighbourhood block filter.

        Parameters
        ----------
        cube
            reference cube or mesh, as-per
            :class:`~ugants.filter.generic.UnstructuredFilterABC`.
        central_fraction
            the weight applied to the centre face value at each point
        neighbours_fraction
            the weight applied to the mean of all neighbouring values at each point

        Examples
        --------
        >>> # block-average-like calculation over nearest neighbours only
        >>> blur_filter = FaceNeighbourhoodFilter(input_cube, central_fraction=0.2, neighbours_fraction=0.8)
        >>> # an edge-detection type operation
        >>> edge_filter = FaceNeighbourhoodFilter(input_cube, central_fraction=1., neighbours_fraction=-1.)
        >>> # a "sharpening" (anti-blur) effect
        >>> sharpen_filter = FaceNeighbourhoodFilter(input_cube, central_fraction=2.0, neighbours_fraction=-1.0)
        >>> # operation
        >>> sharpened_result = sharpen_filter(input_cube)

        Notes
        -----
        * cubes passed to both the constructor and `__call__` methods must have
          ``mesh.location == 'face'``,
        * the mesh must have faces *and* a face-face connectivity.

        """  # noqa: E501
        super().__init__(cube)
        if not self.location == "face":
            message = (
                "A FaceNeighbourhoodFilter can only work with face-located data. "
                f'The provided reference cube, "{cube.name()}", has '
                f"location = {cube.location!r}."
            )
            raise ValueError(message)
        if not cube.mesh.face_face_connectivity:
            message = (
                "A FaceNeighbourhoodFilter requires face connectivity data. "
                f'The provided reference cube, "{cube.name()}", has a mesh with '
                f'an empty "mesh.face_face_connectivity".'
            )
            raise ValueError(message)
        self.central_fraction = central_fraction
        self.neighbours_fraction = neighbours_fraction

    def inner_filter(self, cube, original_standard_name):
        """Inner filter routine : calculate the neighbourhood operation."""
        # First calculate weights for the neighbourhood calculation.
        # Get indices of neighbours of each face : ensuring that the dimensions
        # are (face_index, neighbour_index), and converting to 0-based indices
        # N.B. this can have *different* numbers of neighbours at different points.
        connectivity = self.mesh.face_face_connectivity
        face_neighbours_indices = (
            connectivity.indices_by_location() - connectivity.start_index
        )

        # Replace all "missing" neighbour-indexes with the 'number_of_faces' value,
        # i.e. one more than the maximum valid face index.
        # For the calculation, a non-masked index array is clearer + quicker, and
        # the 'extra' face is used to ensure that missing neighbours contribute zeros.
        number_of_faces = face_neighbours_indices.shape[0]
        face_neighbours_indices = face_neighbours_indices.filled(number_of_faces)

        # Work out the number of neighbours of each face location, for the
        # divide step deducing the mean-over-neighbours.
        # This is actually a slightly odd way of doing it, as it can't discount masked
        # input datapoints from the average.  But it is a good deal simpler.
        face_neighbours_counts = np.count_nonzero(
            face_neighbours_indices != number_of_faces, axis=-1
        )
        # Enforce a minimum count of 1, to avoid divide-by-zeros where a face has no
        # neighbours : this results in a sum over neighbours of 0.0 instead of NaN.
        face_neighbours_counts = face_neighbours_counts.clip(min=1, max=None)

        # Fetch cube data, transposed so mesh is the first dim (makes much simpler)
        face_data = move_one_dimension(cube.data.copy(), cube.mesh_dim(), 0)

        # Add an extra face index to the source data, containing all-zeros
        # This is so that we can index to a "missing" face and there always find "0.0".
        # Make a zero end-column, of shape (1, dim1, dim2 ...), matching the data shape
        # (number_of_faces, dim1, dim2 ...)
        extra_shape = [1, *list(face_data.shape[1:])]
        extra_face_data = np.zeros(extra_shape, dtype=face_data.dtype)
        # N.B. we must use "np.ma.concatenate", since regular "np.concatenate" does not
        # handle masked input correctly.  See https://github.com/numpy/numpy/issues/8881
        # N.B.#2 this means *all* results are masked.  That is not really a problem.
        result_data = np.ma.concatenate((face_data, extra_face_data), axis=0)

        # Construct a sum-over-neighbours at each point.
        neighbour_sums = result_data[face_neighbours_indices].sum(axis=1)
        # ..dividing by (pre-computed) number of neighbours at each point.
        # NOTE: this give results which might be unexpected for masked datapoints : as
        # it divides by a pre-calculated number of neighbours, it treats any *masked*
        # input points as if they were ZEROS.
        n_neighbours_shape = [number_of_faces] + [1] * (cube.ndim - 1)
        n_neighbours = face_neighbours_counts.reshape(n_neighbours_shape)
        neighbours_mean = neighbour_sums / n_neighbours

        # Scale and add to get the result array.
        result_data = (
            self.central_fraction * face_data
            + self.neighbours_fraction * neighbours_mean
        )

        # Assign back to cube.
        cube.data = move_one_dimension(result_data, 0, cube.mesh_dim())
        return cube
