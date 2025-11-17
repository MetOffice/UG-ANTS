# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris.coords
import iris.cube
import numpy as np
import pytest
from ugants.analysis.fill import fill_cube
from ugants.tests import stock


@pytest.fixture()
def source_cube_1d():
    """Create a mesh cube with one dimension.

    The mesh is equivalent to a 3x3 lat-lon grid:
    2 5 8
    1 4 7
    0 3 6
    """
    cube = stock.regular_lat_lon_mesh_cube(
        min_lon=0.0, max_lon=10.0, min_lat=0.0, max_lat=10.0, shape=(3, 3)
    )
    return cube


@pytest.fixture()
def source_cube_2d(source_cube_1d):
    """Create a mesh cube with two dimensions: one horizontal and one vertical.

    The mesh is equivalent to a 3x3 lat-lon grid:
    level 0   level 1
    2 5 8     11 14 17
    1 4 7     10 13 16
    0 3 6      9 12 15
    """
    levels_coord = iris.coords.DimCoord([0, 1], long_name="level")
    level_0_cube = source_cube_1d.copy()
    level_0_cube.add_aux_coord(levels_coord[0])
    level_1_cube = source_cube_1d.copy()
    level_1_cube.add_aux_coord(levels_coord[1])
    level_1_cube.data += 9
    cube = iris.cube.CubeList([level_0_cube, level_1_cube]).merge_cube()
    return cube


class TestExceptions:
    def test_invalid_fill_method(self):
        """Test that an error is raised when an invalid fill method is passed."""
        source = stock.mesh_cube()
        expected_msg = (
            "Unrecognised fill method: 'foo'. "
            r"Supported fill methods are \('kdtree',\)"
        )
        with pytest.raises(ValueError, match=expected_msg):
            fill_cube(source, method="foo")


class TestStrict:
    def test_1d_cube(self, source_cube_1d):
        """Test that a missing point is filled in a 1D cube.

        Before fill:
        2 5 8
        X 4 7
        0 3 6

        After fill:
        2 5 8
        4 4 7
        0 3 6
        """
        source_cube_1d.data[1] = np.ma.masked
        expected_filled = source_cube_1d.copy()
        expected_filled.data[1] = 4
        filled = fill_cube(source_cube_1d)
        assert expected_filled == filled

    def test_2d_cube_consistent_masks(self, source_cube_2d):
        """Test that the missing points are filled in a 2D cube.

        The same fill index should be used across both levels.

        Before fill:
        2 5 8   11 14 17
        X 4 7    X 13 16
        0 3 6    9 12 15

        After fill:
        2 5 8   11 14 17
        4 4 7   13 13 16
        0 3 6    9 12 15
        """
        source_cube_2d.data[:, 1] = np.ma.masked
        expected_filled = source_cube_2d.copy()
        expected_filled.data[0, 1] = 4
        expected_filled.data[1, 1] = 13
        filled = fill_cube(source_cube_2d)
        assert expected_filled == filled

    def test_2d_cube_inconsistent_masks_error(self, source_cube_2d):
        """Test that an error is raised if the two levels have inconsistent masks.

        Before fill:
        2 5 8   11 14 17
        X 4 7   10 13 16
        0 3 6    X 12 15
        """
        source_cube_2d.data[0, 1] = np.ma.masked
        source_cube_2d.data[1, 0] = np.ma.masked
        expected_msg = (
            "Cannot fill the provided cube as it does not have the same missing "
            "data as the source."
        )
        with pytest.raises(ValueError, match=expected_msg):
            fill_cube(source_cube_2d)


class TestLenient:
    def test_2d_cube_inconsistent_masks_error(self, source_cube_2d):
        """Test that a missing points are filled in a 2D cube.

        A different fill index should be used across both levels.

        Before fill:
        2 5 8   11 14 17
        X 4 7   10 13 16
        0 3 6    X 12 15

        After fill:
        2 5 8   11 14 17
        4 4 7   10 13 16
        0 3 6   12 12 15
        """
        source_cube_2d.data[0, 1] = np.ma.masked
        source_cube_2d.data[1, 0] = np.ma.masked
        expected_filled = source_cube_2d.copy()
        expected_filled.data[0, 1] = 4
        expected_filled.data[1, 0] = 12
        filled = fill_cube(source_cube_2d, strict=False)
        np.testing.assert_array_equal(filled.data, expected_filled.data)
        assert expected_filled == filled
