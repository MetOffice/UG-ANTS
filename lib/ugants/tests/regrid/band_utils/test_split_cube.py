# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import re

import iris
import iris.coords
import iris.cube
import numpy as np
import pytest

from ugants.regrid.band_utils import (
    split_cube,
)
from ugants.tests import get_data_path


@pytest.fixture()
def non_ugrid_cube():
    input_filepath = get_data_path("non_ugrid_data.nc")
    cube = iris.load_cube(input_filepath)
    return cube


class TestSplitCube:
    def test_latitude_split_and_concatenate_round_trip(self, non_ugrid_cube):
        cubes = split_cube(non_ugrid_cube, 10, axis="y")
        result = cubes.concatenate_cube()
        assert non_ugrid_cube == result

    def test_latitude_split_by_default(self, non_ugrid_cube):
        expected_cubes = split_cube(non_ugrid_cube, 10, axis="y")
        actual_cubes = split_cube(non_ugrid_cube, 10)

        for actual, expected in zip(actual_cubes, expected_cubes, strict=True):
            assert actual == expected

    def test_longitude_split_and_concatenate_round_trip(self, non_ugrid_cube):
        expected = non_ugrid_cube
        # Pieces of split cube are not cirular, so we do not expect the
        # concatenated result to have the circular attribute set to True
        # either:
        expected.coord(axis="x").circular = False

        cubes = split_cube(non_ugrid_cube, 10, axis="x")
        actual = cubes.concatenate_cube()
        assert actual == expected

    def test_split_into_correct_number_of_cubes(self, non_ugrid_cube):
        expected = 10

        cubes = split_cube(non_ugrid_cube, 10)
        actual = len(cubes)

        assert actual == expected

    def test_latitude_split_is_latitude(self, non_ugrid_cube):
        global_latitude = non_ugrid_cube.coord(axis="y")
        global_latitude_range = np.max(global_latitude.points) - np.min(
            global_latitude.points
        )

        cubes = split_cube(non_ugrid_cube, 10, axis="y")
        for cube in cubes:
            cube_latitude = cube.coord(axis="y")
            cube_latitude_range = np.max(cube_latitude.points) - np.min(
                cube_latitude.points
            )
            assert cube_latitude_range < global_latitude_range

    def test_longitude_split_is_longitude(self, non_ugrid_cube):
        global_longitude = non_ugrid_cube.coord(axis="x")
        global_longitude_range = np.max(global_longitude.points) - np.min(
            global_longitude.points
        )

        cubes = split_cube(non_ugrid_cube, 10, axis="x")
        for cube in cubes:
            cube_longitude = cube.coord(axis="x")
            cube_longitude_range = np.max(cube_longitude.points) - np.min(
                cube_longitude.points
            )
            assert cube_longitude_range < global_longitude_range

    def test_raise_error_for_invalid_axis(self, non_ugrid_cube):
        with pytest.raises(
            ValueError, match=re.escape('Axis must be one of "X" or "Y".  Got: "foo".')
        ):
            split_cube(non_ugrid_cube, 10, axis="foo")

    def test_raise_error_for_multidimensional_input(self, non_ugrid_cube):
        cubes = iris.cube.CubeList()
        cubes.append(iris.util.new_axis(non_ugrid_cube.copy()))
        cubes.append(iris.util.new_axis(non_ugrid_cube.copy()))
        cubes[0].add_dim_coord(iris.coords.DimCoord([0], standard_name="depth"), 0)
        cubes[1].add_dim_coord(iris.coords.DimCoord([1], standard_name="depth"), 0)
        cube = cubes.concatenate_cube()
        assert len(cube.shape) == 3

        with pytest.raises(
            ValueError,
            match=re.escape("Input cube must be 2 dimensional.  Got 3 dimensions."),
        ):
            split_cube(cube, 10)
