# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import re
from unittest.mock import Mock

import numpy as np
import pytest
from iris.coords import AuxCoord

from ugants.regrid.band_utils import (
    cube_subset_latitude_bounds,
)


class TestLatitudeSubsetBounds:
    def test_coord_decreasing_same_range(self):
        mesh = Mock()
        mesh.node_coords.node_y.points = np.array([+90, 0.0])
        coord = AuxCoord(np.linspace(+89.5, +0.5, num=90))
        # Bounds for coord will be [[90.0, 89.0], [89.0, 88.0], ...]
        # Collapsed bounds will be [0.0, 90.0]
        coord.guess_bounds()
        result = cube_subset_latitude_bounds(mesh, coord, fraction=0.0)
        expected = (0.0, 90.0)
        assert expected == result

    def test_coord_increasing_same_range(self):
        mesh = Mock()
        mesh.node_coords.node_y.points = np.array([-90.0, 0.0])
        coord = AuxCoord(np.linspace(-89.5, -0.5, num=90))
        # Bounds for coord will be [[-90.0, -89.0], [-89.0, -88.0], ...]
        # Collapsed bounds will be [-90.0, 0.0]
        coord.guess_bounds()
        result = cube_subset_latitude_bounds(mesh, coord, fraction=0.0)
        expected = (-90.0, 0.0)
        assert expected == result

    def test_coord_decreasing_subset_range(self):
        mesh = Mock()
        mesh.node_coords.node_y.points = np.array([+90, +44.5])
        coord = AuxCoord(np.linspace(+89.5, +0.5, num=90))
        # Bounds for coord will be [[90.0, 89.0], [89.0, 88.0], ...]
        # Collapsed bounds will be [0.0, 90.0]
        coord.guess_bounds()
        result = cube_subset_latitude_bounds(mesh, coord, fraction=0.0)
        # mesh minimum (44.5) will be within bounds [45.0, 44.0].
        expected = (+44.0, +90)
        assert expected == result

    def test_coord_increasing_subset_range(self):
        mesh = Mock()
        mesh.node_coords.node_y.points = np.array([-90.0, -44.5])
        coord = AuxCoord(np.linspace(-89.5, -0.5, num=90))
        # Bounds for coord will be [[-90.0, -89.0], [-89.0, -88.0], ...]
        # Collapsed bounds will be [-90.0, 0.0]
        coord.guess_bounds()
        result = cube_subset_latitude_bounds(mesh, coord, fraction=0.0)
        # mesh minimum (44.5) will be within bounds [-45.0, -44.0].
        expected = (-90.0, -44.0)
        assert expected == result

    def test_coord_fraction(self):
        mesh = Mock()
        mesh.node_coords.node_y.points = np.array([0.0, +25.0])
        coord = AuxCoord(np.linspace(2.5, 47.5, num=10))
        # Bounds for coord will be [[0.0, 5.0], [5.0, 10.0], ...]
        # Collapsed bounds will be [0.0, 50.0]
        coord.guess_bounds()
        result = cube_subset_latitude_bounds(mesh, coord, fraction=0.4)
        # mesh minimum (0.0) will be within bounds [0.0, 5.0] index 0
        # mesh maximum (25.0) will be within bounds [20, 25] index 4
        # total rows spanned is 5, when multiplied by fraction gives 2
        # each row spans 5 degrees, giving an extra 10 degrees padding.
        expected = (0, 35.0)
        assert expected == result

    def test_enclosed_error(self):
        mesh = Mock()
        mesh.node_coords.node_y.points = np.array([+90, 0.0])
        coord = AuxCoord(np.linspace(-44.5, +44.5, num=90))

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Coord bounds (-45.0, 45.0) do not enclose mesh bounds (0.0, 90.0)."
            ),
        ):
            cube_subset_latitude_bounds(mesh, coord, fraction=0.0)
