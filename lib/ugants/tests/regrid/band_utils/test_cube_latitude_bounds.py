# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris
import pytest
from iris import load_cube

from ugants.regrid.band_utils import cube_latitude_bounds
from ugants.tests import get_data_path


@pytest.fixture()
def non_ugrid_cube():
    input_filepath = get_data_path("non_ugrid_data.nc")
    cube = load_cube(input_filepath)
    return cube


class TestCubeLatitudeBounds:
    def test_expected_bounds(self, non_ugrid_cube):
        expected = (-90.0, 90.0)
        result = cube_latitude_bounds(non_ugrid_cube)
        assert expected == result

    def test_no_mutation(self, non_ugrid_cube):
        cube_copy = non_ugrid_cube.copy()
        cube_latitude_bounds(non_ugrid_cube)
        assert non_ugrid_cube == cube_copy

    def test_bounds_always_monotonically_increase(self, non_ugrid_cube):
        expected = (-90.0, 90.0)
        cube = iris.util.reverse(non_ugrid_cube, "latitude")
        result = cube_latitude_bounds(cube)
        assert expected == result
