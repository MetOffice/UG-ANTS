# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Utility functions for handling UGrid cubes."""

import os
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import dask
import iris.cube
import numpy as np

from ugants.exceptions import PROVISIONAL_WARNING_MESSAGE, ProvisionalWarning


def as_cubelist(cubes):
    """
    Return a CubeList, irrespective of whether a Cube or a CubeList has been provided.

    Parameters
    ----------
    cubes : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`

    Raises
    ------
    TypeError
        If provided with anything other than a
        :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`

    Returns
    -------
    :class:`iris.cube.CubeList`
    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])
    elif not isinstance(cubes, iris.cube.CubeList):
        raise TypeError(
            f"Expected iris.cube.Cube or iris.cube.CubeList, got {type(cubes)}"
        )
    return cubes


def is_ugrid(cube):
    """Check if the provided cube contains UGrid data.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        Cube to check.

    Returns
    -------
    bool
    """
    return cube.mesh is not None


def get_connectivity_indices(cube, connectivity):
    """Get the connectivity indices array for a cube's mesh.

    The ``start_index`` is subtracted from the connectivity
    array, so that the returned array is guaranteed to have
    zero-based indexing. For more information about zero-
    and one-based indexing, see the `UGRID conventions
    <http://ugrid-conventions.github.io/ugrid-conventions/#zero-or-one-based-indexing>`_.

    Parameters
    ----------
    cube : iris.cube.Cube
        The cube from which to extract the connectivity indices.
    connectivity : str
        The connectivity to extract.

    Returns
    -------
    connectivity_indices : numpy.ndarray
        The cube's zero-indexed connectivity array.
    """
    connectivity = cube.mesh.connectivity(cf_role=connectivity)
    connectivity_indices = connectivity.indices - connectivity.start_index
    return connectivity_indices


def prepare_for_save(cubes: iris.cube.Cube | iris.cube.CubeList):
    """Add or amend attributes on a cube or cubes prior to saving to disk.

    Each cube is updated so that its history attribute acquires a new entry,
    specifying the operation performed by the invoked application.
    A history entry records a timestamp and the application (i.e. command-line) name and
    arguments of the current operation.
    History "entries" are delimited by newlines: the most recent appears first.

    The `CF Conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#description-of-file-contents>`_
    provide further description of the history attribute.

    A "ugants_status" attribute is added to indicate whether the version of UG-ANTS is
    stable.

    A "suite_provenance" attribute is added to provide information on the suite
    (if applicable) taken from the ``SUITE_PROVENANCE`` environment variable.

    Note
    ----
    A copy of the provided cube(s) is returned,
    the operation is *not* performed in place.

    Parameters
    ----------
    cubes : iris.cube.Cube | iris.cube.CubeList
        Cube(s) to be saved.

    Returns
    -------
    iris.cube.Cube | iris.cube.CubeList
        A copy of the cube(s) with relevant metadata updated.
        If a single :class:`~iris.cube.Cube` is provided, then a single
        :class:`~iris.cube.Cube` will be returned.
        If a :class:`~iris.cube.CubeList` is provided, then a
        :class:`~iris.cube.CubeList` will be returned.

    """
    updated_cubes = iris.cube.CubeList()
    operation_entry_str = _create_history_entry()
    for cube in as_cubelist(cubes):
        updated_cube = _update_single_cube_history(cube.copy(), operation_entry_str)
        updated_cube.attributes["ugants_status"] = PROVISIONAL_WARNING_MESSAGE
        updated_cube.attributes["suite_provenance"] = os.environ.get(
            "SUITE_PROVENANCE", "None"
        )
        updated_cubes.append(updated_cube)
    warnings.warn(PROVISIONAL_WARNING_MESSAGE, ProvisionalWarning, stacklevel=1)
    if isinstance(cubes, iris.cube.Cube):
        updated_cubes = updated_cubes[0]
    return updated_cubes


def _create_history_entry():
    """Create a new entry to be appended to an existing history attribute.

    The new history entry is a timestamped record of the command that was run
    to create or modify the cube.
    """
    date = datetime.today().replace(microsecond=0)
    date_str = date.isoformat()
    app_path, app_args = sys.argv[0], sys.argv[1:]
    app_name = Path(app_path).name
    app_args.insert(0, app_name)
    app_str = " ".join(app_args)
    operation_entry_str = f"{date_str}: {app_str}"
    return operation_entry_str


def _update_single_cube_history(cube: iris.cube.Cube, operation_entry_str: str):
    cube_history_attr = cube.attributes.get("history")
    if isinstance(cube_history_attr, str):
        new_history_str = f"{operation_entry_str}\n{cube_history_attr}"
    elif cube_history_attr is None:
        new_history_str = operation_entry_str
    else:
        raise TypeError(
            f"Cube '{cube.name()}' has a 'history' attribute of non-string "
            f"type : {cube_history_attr!r}."
        )
    updated_cube = cube.copy()
    updated_cube.attributes["history"] = new_history_str
    return updated_cube


class Stencil:
    """Get indices of all cells in the neighbourhood of a given central cell.

    The algorithm follows that outlined in section 4.1 of `this paper
    <https://doi.org/10.5194/gmd-16-1265-2023>`_.

    Examples
    --------
    The following table shows three panels of the C4 cubed sphere, which is used here
    to illustrate creating stencils on an unstructured grid. The numbers in the table
    represent the cell indices. In the example below, cell 0 neighbours cells 1, 4, 51,
    and 64. Cell 50 neighbours cells 49, 51, 54, and 65.

    +---+---+---+---+---+---+---+---+
    |               |67 |71 |75 |79 |
    +               +---+---+---+---+
    |               |66 |70 |74 |78 |
    +               +---+---+---+---+
    |               |65 |69 |73 |77 |
    +               +---+---+---+---+
    |               |64 |68 |72 |76 |
    +---+---+---+---+---+---+---+---+
    |48 |49 |50 |51 |0  |1  |2  |3  |
    +---+---+---+---+---+---+---+---+
    |52 |53 |54 |55 |4  |5  |6  |7  |
    +---+---+---+---+---+---+---+---+
    |56 |57 |58 |59 |8  |9  |10 |11 |
    +---+---+---+---+---+---+---+---+
    |60 |61 |62 |63 |12 |13 |14 |15 |
    +---+---+---+---+---+---+---+---+

    By default the stencil will only include the four immediate neighbours,
    plus the central cell itself:

    >>> stencil = Stencil(cube)
    >>> stencil[0]
    [0, 64, 1, 51, 4]
    >>> stencil[10]
    [6, 9, 10, 11, 14]

    Negative indices are permitted, and will be interpreted in the usual Python way,
    i.e. for a positive integer ``M``, the index ``-M`` will be replaced with ``N - M``
    where ``N`` is the length of the array. For example, index ``-1`` refers to the
    final element of the array, ``-2`` the penultimate element, etc. If the provided
    central index is negative, then it will appear in the
    returned list as its positive counterpart.

    >>> cube.shape
    (96,)
    >>> stencil[-96]
    [0, 64, 1, 51, 4]
    >>> stencil[-86]
    [6, 9, 10, 11, 14]

    Passing ``iterations=2`` generates an extended stencil with the eight
    surrounding cells, or seven if the central cell is on the corner of a face:

    >>> extended_stencil = Stencil(cube, iterations=2)
    >>> extended_stencil[0]
    [0, 64, 1, 51, 4, 68, 5, 55]
    >>> extended_stencil[10]
    [5, 6, 7, 9, 10, 11, 13, 14, 15]

    Further iterations can be specified if a larger neighbourhood is required:

    >>> third_iteration_stencil = Stencil(cube, iterations=3)
    >>> third_iteration_stencil[0]
    [0, 64, 1, 65, 4, 68, 5, 2, 69, 6, 8, 72, 9, 50, 51, 54, 55, 59]
    >>> third_iteration_stencil[10]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 80, 20, 84, 24, 88, 28]

    """

    def __init__(self, cube, iterations=1):
        """Generate a stencil of a given size on a cube.

        Parameters
        ----------
        cube : iris.cube.Cube
            Cube on which to generate stencil.
        iterations : :obj:`int`, optional
            The number of iterations for which to run the stencil generation algorithm.

            * If ``1`` (the default), use only the four immediate neighbours
              around the central cell.
            * If ``2``, use the extended neighbourhood by filling the corner
              cells, i.e. those which are connected to two cells in the immediate
              neighbourhood.

        Raises
        ------
        TypeError
            If the provided ``iterations`` is not an :obj:`int`.
        ValueError
            If the provided ``iterations`` is not a positive integer.
        """
        if not isinstance(iterations, int):
            raise TypeError(f"iterations must be an int, got {type(iterations)}.")
        if iterations < 1:
            raise ValueError(
                f"iterations must be a positive integer, got {iterations}."
            )

        # Get the zero-indexed connectivity array
        self.connectivity_indices = get_connectivity_indices(
            cube,
            "face_face_connectivity",
        )
        self._num_faces = cube.shape[0]

        self.iterations = iterations

    def __getitem__(self, central_cell_index):
        """Retrieve the indices of the neighbourhood of ``central_cell_index``.

        Parameters
        ----------
        central_cell_index : int
            Index of the central cell around which the stencil is defined. Negative
            indices are permitted, and will be interpreted in the usual Python way, i.e.
            index ``-1`` refers to the final element of the array, ``-2`` the
            penultimate element, etc.

        Returns
        -------
        list[int]
            All indices in the stencil, including the central cell index.
            The list is unordered, and contains only positive indices.

        Note
        ----
        Negative indices are permitted, and will be interpreted in the usual Python way,
        i.e. index ``-1`` refers to the final element of the array, ``-2`` the
        penultimate element, etc. If the provided central index is negative, then it
        will appear in the returned list as its positive counterpart, e.g. ``-1`` will
        be replaced with ``N - 1`` where ``N`` is the number of faces of the cube.
        """
        # Allow negative indices as long as they are greater than -len(data)
        if (central_cell_index < -self._num_faces) or (
            central_cell_index >= self._num_faces
        ):
            raise IndexError(
                f"Cannot index face {central_cell_index} for array "
                f"of length {self._num_faces}."
            )

        # Rebase to convert negative index to equivalent positive index.
        # The reason for this is to avoid double counting in the neighbourhood,
        # e.g. -1 and 95 would be counted as separate indices for a C4 cubed-sphere.
        central_cell_index = central_cell_index % self._num_faces

        # Initialise neighbourhood with central cell only
        neighbourhood = {central_cell_index}
        for _ in range(self.iterations):
            # get faces connected to the current neighbourhood
            connected_faces = [
                face
                for face in self.connectivity_indices[list(neighbourhood)].flatten()
                if face not in neighbourhood
            ]
            # count how many faces connect to each of these outer faces
            face_counts = Counter(connected_faces)

            # get the cells that are the most connected to the current neighbourhood
            new_faces = {
                face
                for face, count in face_counts.items()
                if count == max(face_counts.values())
            }
            neighbourhood |= new_faces

        return list(neighbourhood)


def align_mask(cube_input):
    """
    Adjust a cube's mask so that it is of the same shape as the associated data.

    Tests the input to see if it should be handled as a cube or cubelist and
    uses ```_fix_cube_mask(cube)``` to carry out the work of adjusting the mask(s).

    Parameters
    ----------
    cube_input : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`

    Returns
    -------
    : None
        In-place operation.

    """
    if isinstance(cube_input, iris.cube.CubeList):
        for cube in cube_input:
            _expand_cube_mask(cube)
    else:
        _expand_cube_mask(cube_input)


def _expand_cube_mask(cube):
    """
    If the input cube has no mask, this routine returns an mask array of
    False values that matches the shape of the input core data array.
    It is designed to address cases where a single False numpy boolean
    is being returned as a mask rather than a data sized array of False
    values. It maintains unrealised data if input is lazy.
    """ #noqa 205
    lazy = cube.has_lazy_data()
    cube_core_data = cube.core_data()
    if lazy:
        mask_values = dask.array.ma.getmaskarray(cube_core_data)
    else:
        mask_values = np.ma.getmask(cube_core_data)

    if cube_core_data.shape != mask_values.shape:
        if lazy:
            cube.data = dask.array.ma.masked_array(
                cube_core_data, mask=np.zeros(cube_core_data.shape, dtype=bool)
            )
        else:
            cube.data = np.ma.masked_array(
                cube_core_data, mask=np.zeros(cube_core_data.shape, dtype=bool)
            )
    return
