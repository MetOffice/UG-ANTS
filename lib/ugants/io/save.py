# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Package for saving data to netcdf-ugrid files.

The routines also perform some standard checks on the source data to ensure that is
has expected properties, such as all cubes being unstructured.
"""
import os
import warnings
from pathlib import Path

from iris.experimental.ugrid import Mesh, save_mesh
from iris.fileformats.netcdf import save as save_netcdf

from ugants.exceptions import PROVISIONAL_WARNING_MESSAGE, ProvisionalWarning
from ugants.utils.cube import (
    _create_history_entry,
    as_cubelist,
    is_ugrid,
    prepare_for_save,
)


def ugrid(cubes, filepath, **save_kwargs):
    """
    Save one or more unstructured Iris cubes to a netCDF4 UGRID-type file.

    The resultant file will also be provided with two additional attributes.

    * ``ugants_status`` - Indicating whether the version of UG-ANTS is stable.
    * ``suite_provenance`` - Information on the suite (if applicable) taken from
      the ``SUITE_PROVENANCE`` environment variable.

    Parameters
    ----------
    cubes : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
        One or more cubes to be saved.
        All cubes must be unstructured, with the *same* mesh.
    filepath : str or :class:`pathlib.Path`
        Creation path of a single netCDF file.
        Must have the extension ".nc".
    **save_kwargs : dict
        Any additional control keywords,
        passed to :func:`iris.fileformats.netcdf.save`.
    """
    filepath = Path(filepath)

    _check_filepath_extension(filepath)

    # Convert single cube to iterable of cubes, for simplicity
    cubes = as_cubelist(cubes)
    if any(not is_ugrid(cube) for cube in cubes):
        raise ValueError("Provided cubes are not all unstructured.")

    if any(cube.mesh != cubes[0].mesh for cube in cubes):
        raise ValueError("Provided cubes are not all on the same mesh.")

    output_cubes = prepare_for_save(cubes)

    save_netcdf(output_cubes, filepath, **save_kwargs)


def mesh(mesh: Mesh, filepath: str | Path, **save_kwargs):
    """
    Save one unstructured mesh to a netCDF4 UGRID-type file.

    If a history attribute already exists on the provided mesh, then a new history entry
    will be prepended. If the provided mesh has no history, then one will be added.

    The resultant file will also be provided with two additional attributes.

    * ``ugants_status`` - Indicating whether the version of UG-ANTS is stable.
    * ``suite_provenance`` - Information on the suite (if applicable) taken from
      the ``SUITE_PROVENANCE`` environment variable.

    Note
    ----
    The attributes will be added as local attributes to the mesh topology variable
    of the mesh. Iris 3.9 does not support adding global variables to meshes.
    See Iris issue `#6085 <https://github.com/SciTools/iris/issues/6085>`_.


    Warning
    -------
    This function modifies the mesh **in place** by adding/modifying attributes.
    Iris currently does not have support for copying meshes, see Iris issue
    `#5982 <https://github.com/SciTools/iris/issues/5982>`_.

    Parameters
    ----------
    mesh : :class:`~iris.experimental.ugrid.Mesh`
        A mesh to be saved.
    filepath : str or :class:`pathlib.Path`
        Creation path of a single netCDF file.
        Must have the extension ".nc".
    **save_kwargs : dict
        Any additional control keywords,
        passed to :func:`iris.experimental.ugrid.save_mesh`.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Expected mesh, got {type(mesh)}.")
    filepath = Path(filepath)
    _check_filepath_extension(filepath)

    operation_entry_str = _create_history_entry()
    mesh.attributes["history"] = _create_new_mesh_history(mesh, operation_entry_str)
    mesh.attributes["ugants_status"] = PROVISIONAL_WARNING_MESSAGE
    mesh.attributes["suite_provenance"] = os.environ.get("SUITE_PROVENANCE", "None")
    warnings.warn(PROVISIONAL_WARNING_MESSAGE, ProvisionalWarning, stacklevel=1)
    save_mesh(mesh, filepath, **save_kwargs)


def _check_filepath_extension(filepath: Path):
    if not filepath.name.endswith(".nc"):
        message = (
            f'Provided output filepath, "{filepath!s}", does not have extension ".nc"'
        )
        raise ValueError(message)


def _create_new_mesh_history(mesh: Mesh, operation_entry_str: str):
    """Create a new history entry for a mesh using operation entry string."""
    existing_history_attr = mesh.attributes.get("history")
    if isinstance(existing_history_attr, str):
        new_history_str = f"{operation_entry_str}\n{existing_history_attr}"
    elif existing_history_attr is None:
        new_history_str = operation_entry_str
    else:
        raise TypeError(
            f"Mesh '{mesh.name()}' has a 'history' attribute of non-string "
            f"type : {existing_history_attr!r}."
        )
    return new_history_str
