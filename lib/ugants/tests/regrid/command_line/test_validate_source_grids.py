# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris.coords
import iris.cube
import pytest

from ugants.io import load
from ugants.regrid.command_line import _validate_source_grids
from ugants.tests import get_data_path


@pytest.fixture()
def source_cubelist():
    return load.cf(get_data_path("non_ugrid_data.nc"))


class TestSameGrids:
    def test_same_source_grids(self, source_cubelist):
        """Validate a cubelist with two identical cubes."""
        source_cubelist.extend(source_cubelist)
        assert _validate_source_grids(source_cubelist) is None

    def test_same_horizontal_different_vertical(self, source_cubelist):
        """Validate a cubelist with different vertical grids and identical horizontal grids.

        First cube has no vertical levels, only horizontal coordinates. The second cube
        has the same horizontal grid as the first cube, and also has a vertical
        coordinate 'model_level_number'.

        No error should be raised because only the horizontal grid must be the same.
        """  # noqa: E501
        second_cube_0 = source_cubelist[0].copy()
        second_cube_1 = source_cubelist[0].copy()
        vertical_coord = iris.coords.DimCoord([0, 1], "model_level_number")
        second_cube_0.add_aux_coord(vertical_coord[0])
        second_cube_1.add_aux_coord(vertical_coord[1])
        second_cube = iris.cube.CubeList([second_cube_0, second_cube_1]).merge_cube()
        source_cubelist.append(second_cube)
        assert _validate_source_grids(source_cubelist) is None


class TestDifferentGrids:
    """Tests that a ValueError is raised when source cubes have different grids."""

    def test_different_source_grid_x(self, source_cubelist):
        """Test that a ValueError is raised when cubes have different x coords."""
        second_cube = source_cubelist[0].copy()
        # slice on x dimension, so second cube has different grid to first
        second_cube = second_cube[:, :20]
        source_cubelist.append(second_cube)
        with pytest.raises(
            ValueError, match="Not all source cubes have the same horizontal grid."
        ):
            _validate_source_grids(source_cubelist)

    def test_different_source_grid_y(self, source_cubelist):
        """Test that a ValueError is raised when cubes have different y coords."""
        second_cube = source_cubelist[0].copy()
        # slice on y dimension, so second cube has different grid to first
        second_cube = second_cube[:20, :]
        source_cubelist.append(second_cube)
        with pytest.raises(
            ValueError, match="Not all source cubes have the same horizontal grid."
        ):
            _validate_source_grids(source_cubelist)

    def test_different_source_grid_both(self, source_cubelist):
        """Test that a ValueError is raised when cubes have different x and y coords."""
        second_cube = source_cubelist[0].copy()
        second_cube = second_cube[:20, :30]
        source_cubelist.append(second_cube)
        with pytest.raises(
            ValueError, match="Not all source cubes have the same horizontal grid."
        ):
            _validate_source_grids(source_cubelist)
