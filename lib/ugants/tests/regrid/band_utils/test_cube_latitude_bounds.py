# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris
import pytest

from ugants.regrid.band_utils import cube_latitude_bounds
from ugants.tests.stock import regular_grid_global_cube


@pytest.fixture()
def non_ugrid_cube():
    return regular_grid_global_cube(144, 192)


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
