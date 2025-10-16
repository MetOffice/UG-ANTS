# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris.cube
import pytest
from ugants.tests.stock import mesh_cube
from ugants.utils.cube import as_cubelist


def test_cube():
    cube = mesh_cube()
    expected = iris.cube.CubeList([cube])
    actual = as_cubelist(cube)
    assert actual == expected


def test_cubelist():
    cubelist = iris.cube.CubeList([mesh_cube()])
    expected = iris.cube.CubeList([mesh_cube()])
    actual = as_cubelist(cubelist)
    assert actual == expected


def test_list_of_cubes():
    list_of_cubes = [mesh_cube()]
    expected_message = (
        "Expected iris.cube.Cube or iris.cube.CubeList, got <class 'list'>"
    )
    with pytest.raises(TypeError) as error:
        as_cubelist(list_of_cubes)
    assert str(error.value) == expected_message
