# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import pytest

from ugants.regrid.band_utils import (
    constrain_source_cube_latitude,
)
from ugants.tests.stock import regular_grid_global_cube


@pytest.fixture()
def non_ugrid_cube():
    return regular_grid_global_cube(144, 192)


class TestConstrainSourceCubeLatitude:
    def test_constrain_source_cube_no_padding(self, non_ugrid_cube):
        bounds = (-90, 0)
        cube = constrain_source_cube_latitude(non_ugrid_cube, bounds)
        max_cell_index = cube.coord("latitude").nearest_neighbour_index(max(bounds))
        result = cube.coord("latitude").bounds[max_cell_index]
        assert max(result) == 0.0

    def test_constrain_source_cube_with_padding(self, non_ugrid_cube):
        bounds = (-90, 3)
        padding = 1.25
        cube = constrain_source_cube_latitude(non_ugrid_cube, bounds, padding=padding)
        max_cell_index = cube.coord("latitude").nearest_neighbour_index(max(bounds))
        cell_bounds = cube.coord("latitude").bounds[max_cell_index]
        assert max(cell_bounds) == 3.75
