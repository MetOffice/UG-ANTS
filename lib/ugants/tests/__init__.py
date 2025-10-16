# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Tests for ugants module.

Note: we provide tests unit-test-style for individual software components, arranged in
test sub-modules which are named according to the import path of the component.
Most tests are, nevertheless, "integration tests" in form -- i.e. they mostly exercise
the full call stack, calling on related components in the usual way instead of "mocking"
them out.
"""
from pathlib import Path

import iris

_TESTS_PATH = Path(__file__).parent
_RESOURCE_PATH = _TESTS_PATH / "resources"


def get_data_path(file_name: str) -> str:
    """
    Get the full file path for test data file located in resources directory.

    Parameters
    ----------
    file_name : str
        Name of file to extract file path for,

    Returns
    -------
    str
        File path to the specified test data,

    Raises
    ------
    ValueError
        If the specified file does not exist.
    """
    data_path = _RESOURCE_PATH / file_name
    if data_path.exists():
        return str(data_path)
    else:
        raise ValueError(f"Provided filepath does not exist: {data_path}")


def add_axis(original_cube, name, axis=""):
    """Return a new cube with an additional coordinate named ``name``.

    New coordinate is an :class:`iris.coords.DimCoord`, and is added as a
    leading dimension.

    An optional axis can be passed that sets metadata on the added coordinate so that
    it can be identified using ``cube.coord(axis="z")``, for example.

    Parameters
    ----------
    original_cube : iris.cube.Cube
        Cube to add a new axis to
    name : str
        Name of coordinate associated with new axis
    axis : str, optional
        Axis type, either "Z" or "T".

    Returns
    -------
    iris.cube.Cube


    """
    extended_cube = iris.util.new_axis(original_cube)
    cubes = iris.cube.CubeList([extended_cube, extended_cube.copy()])
    for index, cube in enumerate(cubes):
        coordinate = iris.coords.DimCoord([index])
        # rename() ensures name goes to standard name if it's a valid standard
        # name, or long name otherwise.
        coordinate.rename(name)
        match axis.lower():
            case "z":
                coordinate.attributes["positive"] = "up"
            case "t":
                coordinate.units = "days since 1970-01-01 00:00:00"
            case "":
                pass
            case _:
                raise ValueError(
                    f"Unexpected axis keyword: '{axis}'. Allowed values are 'Z' or 'T'."
                )
        cube.add_dim_coord(coordinate, 0)
    return cubes.concatenate_cube()
