# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import numpy as np
import pytest

from ugants.io.load import ugrid as load_ugrid
from ugants.regrid.band_utils import (
    find_cell_centres_within_latitude_bounds,
    mesh_to_cube,
)
from ugants.tests import get_data_path
from ugants.tests.stock import regular_lat_lon_mesh


@pytest.fixture()
def c4_cube():
    input_filepath = get_data_path("data_C4.nc")
    cubelist = load_ugrid(input_filepath, "sample_data")
    cube = cubelist[0]
    return cube


class TestFindCellCentresWithinLatitudeBounds:
    def test_latitude(self, c4_cube):
        bounds = (-90, 0)

        indices = find_cell_centres_within_latitude_bounds(
            c4_cube,
            bounds,
            True,
            True,
        )
        assert c4_cube.data.size // 2 == np.count_nonzero(indices)

    def test_regular_mesh_inclusive(self):
        """Test inclusive functionality.

        The face cell centres of the dummy mesh are simply defined by the
        midpoint of two edges. The two edges chosen must be orthognal to each other.

        0,4---4,4---8,4
         | 2,3 | 6,3 |
        0,2---4,2---8,2
         | 2,1 | 6,1 |
        0,0---4,0---8,0

        """
        latitude_bounds = (0.5, 9.5)
        mesh = regular_lat_lon_mesh(0, 10, 0, 20, (10, 20))
        cube = mesh_to_cube(mesh)
        indices = find_cell_centres_within_latitude_bounds(
            cube,
            latitude_bounds,
            True,
            True,
        )
        assert np.count_nonzero(indices) == 100

    def test_regular_mesh_exclusive(self):
        """Test exclusive functionality.

        See explanation for ``test_regular_mesh_inclusive``.
        """
        latitude_bounds = (0.5, 9.5)
        mesh = regular_lat_lon_mesh(0, 10, 0, 10, (10, 10))
        cube = mesh_to_cube(mesh)
        indices = find_cell_centres_within_latitude_bounds(
            cube,
            latitude_bounds,
            False,
            False,
        )
        assert np.count_nonzero(indices) == 80
