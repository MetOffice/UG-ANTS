# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Load functions for UG-ANTS."""

from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import iris
import iris.cube
import iris.exceptions
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD, Mesh, load_meshes

from ugants.utils.cube import is_ugrid


def ugrid(uris, constraints=None) -> iris.cube.CubeList:
    """Load a UGrid file. Will not load if the file contains regular data.

    Parameters
    ----------
    uris : Any
        Location of file to load. Must be a NetCDF file.
    constraints: Any | None
        One or more iris constraints. Each constraint can be either a string,
        or an instance of :class:`iris.Constraint`. If the constraint is a string
        it will be used to match against cube.name().

    Returns
    -------
    iris.cube.CubeList
        An iris :class:`iris.cube.CubeList` containing the loaded data.

    Raises
    ------
    iris.exceptions.InvalidCubeError
        If the specified file does not contain UGrid data.
    iris.exceptions.InvalidCubeError
        If no data can be found which matches the provided constraint(s).
    iris.exceptions.InvalidCubeError
        If a mesh has been removed from a cube during constrained load.
        This may be caused by constraining on an unstructured dimension.
    """
    with PARSE_UGRID_ON_LOAD.context():
        ugrid_cubelist = iris.load(
            uris, constraints=constraints, callback=_check_for_non_ugrid
        )

    if len(ugrid_cubelist) == 0:
        raise iris.exceptions.InvalidCubeError(
            f"No data found in file(s) '{uris}' matching the "
            f"provided constraint(s) '{constraints}'."
        )

    # By constraining on a horizontal (unstructured) dimension, iris attempts to
    # subset a MeshCoord. This causes the MeshCoords to be converted to AuxCoords,
    # so the resulting cube has no mesh. This will result in non-ugrid data.
    if not all(is_ugrid(cube) for cube in ugrid_cubelist):
        raise iris.exceptions.InvalidCubeError(
            f"Attempting to load UGrid data from '{uris}' with constraint(s) "
            f"'{constraints}' has resulted in non-UGrid data being loaded. "
            "This may be caused by constraining on an unstructured dimension."
        )

    return ugrid_cubelist


def _check_for_non_ugrid(cube: iris.cube.Cube, field, filename):
    """Check if the loaded data is UGrid or not.

    A function to be passed to callback in iris.load.

    Parameters
    ----------
    cube : iris.cube.Cube
        Loaded :class:`iris.cube.CubeList` to be checked
    field : Any
        Only present to satisfy the function signature used by callback.
    filename : Any
        Original file from which the cube data was loaded

    Raises
    ------
    ValueError
        If cube does not contain UGrid data
    """
    if not is_ugrid(cube):
        raise iris.exceptions.InvalidCubeError(
            f"Specified file '{filename}' does not contain UGrid data."
        )


def cf(uris, **kwargs):
    """Load one or more regular data input files.

    Fails if files are identified as being UGRID.

    Parameters
    ----------
    uris : str or :class:`pathlib.Path`
        A string or Path object to a file containing regular data.
        Can also be a list of files.

    Returns
    -------
    cubes : iris.cube.CubeList
        An iris :class:`iris.cube.CubeList`.

    Raises
    ------
    ValueError
        Raised if any input uris are not netCDF files.

    """
    is_netcdf(uris)

    cubes = iris.load(uris, **kwargs)
    for cube in cubes:
        error_if_mesh(cube, uris)

    return cubes


def mesh(
    uri: str | Path, mesh_name: str | None = None
) -> iris.experimental.ugrid.mesh.Mesh:
    """Load a single mesh from a uri.

    If no mesh_name is given, the mesh_name is inferred from the file itself,
    provided there is only one mesh in the file.

    Parameters
    ----------
    uri : str | :class:`pathlib.Path`
        A string or Path to a netCDF file containing the mesh.
    mesh_name : str | None
        The name of the mesh to load.

    Returns
    -------
    iris.experimental.ugrid.mesh.Mesh
        An iris Mesh as provided by :func:iris.experimental.ugrid.load_mesh

    Raises
    ------
    ValueError
        If multiple meshes are present.
        If no meshes are present.
    """
    mesh_list = meshes(str(uri), mesh_name)
    if len(mesh_list) > 1:
        mesh_names = [mesh.var_name for mesh in mesh_list]
        raise ValueError(f"Expected one mesh, found {len(mesh_list)}. {mesh_names}")
    return mesh_list[0]


def meshes(uris: str | Iterable[str], mesh_name: str | None = None) -> list[Mesh]:
    """Load multiple meshes from netCDF file(s).

    Parameters
    ----------
    uris
        Filepath(s) from which to load meshes.
    mesh_name
        The ``var_name`` of the meshes to load.
        If None (the default), all meshes are loaded.

    Returns
    -------
    list[~iris.experimental.ugrid.Mesh]
        A list of meshes loaded from the ``uris`` with ``mesh_name``, if provided.

    Raises
    ------
    ValueError
        If no meshes are present at the given ``uris`` matching the
        ``mesh_name`` (if provided).
    """
    is_netcdf(uris)
    with PARSE_UGRID_ON_LOAD.context():
        mesh_dict = load_meshes(uris, mesh_name)

    error_message = f"No meshes were found at {uris}"
    if mesh_name:
        error_message = f"{error_message} with var_name '{mesh_name}'"
    if not mesh_dict:
        raise ValueError(error_message)

    # Iris' load_meshes returns a dict mapping uri to a list of loaded meshes
    # We will extract just the lists
    return list(chain(*mesh_dict.values()))


def is_netcdf(uris):
    """Check uris point to netCDF files.

    Parameters
    ----------
    uris : str or :class:`pathlib.Path`
        A string or Path object to a file containing regular data.
        Can also be a list of files.

    Raises
    ------
    ValueError
        Raised if any input uris are not netCDF files.
    """
    if isinstance(uris, str | Path):
        uris = [uris]

    uris = [Path(uri) for uri in uris]

    for uri in uris:
        if not uri.suffix == ".nc":
            message = f"Input file '{uri}' must be netCDF."
            raise ValueError(message)


def error_if_mesh(cube, filename):
    """Perform basic sanity checks to prevent accidently loading UGRID data.

    A callback function for use with iris.load.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The iris cube to perform checking on.
    filename : Any
        Original file used to generate the cube.

    Raises
    ------
    ValueError
        If cf_role is defined and is 'mesh_topology'

    """
    if hasattr(cube, "attributes"):
        cf_role = cube.attributes.get("cf_role", None)
        if cf_role == "mesh_topology":
            message = (
                f"Cannot load input file '{filename!s}' as 'cf_role' is '{cf_role}'"
            )
            raise ValueError(message)
