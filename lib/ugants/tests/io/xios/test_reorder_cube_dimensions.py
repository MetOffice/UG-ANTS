# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import itertools

import iris.coords
import pytest
from ugants.io.xios_command_line import reorder_cube_dimensions
from ugants.tests import add_axis
from ugants.tests.stock import four_vertical_levels_cube, mesh_cube


class TestMeshDimensionOnly:
    """Tests for the case where the source cube only has a mesh dimension.

    Cubes should be unchanged by the reorder dimension.
    """

    def test_mesh_only_cube_unchanged(self):
        """Test for the case where the source cube only has a mesh dimension."""
        cube = mesh_cube()
        original_cube = cube.copy()
        reordered_cube = reorder_cube_dimensions(cube)
        assert original_cube == reordered_cube

    def test_mesh_only_cube_with_time_scalar_coord(self):
        """Test for the case where the source cube has a mesh dimension and a time scalar coord."""  # noqa: E501
        cube = mesh_cube()
        time_scalar_coord = iris.coords.AuxCoord(
            [1], standard_name="time", units="days since 1970-01-01 00:00:00"
        )
        cube.add_aux_coord(time_scalar_coord)
        original_cube = cube.copy()
        reordered_cube = reorder_cube_dimensions(cube)
        assert original_cube == reordered_cube

    def test_mesh_only_cube_with_vertical_scalar_coord(self):
        """Test for the case where the source cube has a mesh dimension and a vertical scalar coord."""  # noqa: E501
        cube = mesh_cube()
        vertical_scalar_coord = iris.coords.AuxCoord(
            [1],
            standard_name="model_level_number",
            units="1",
            attributes={"positive": "up"},
        )
        cube.add_aux_coord(vertical_scalar_coord)
        original_cube = cube.copy()
        reordered_cube = reorder_cube_dimensions(cube)
        assert original_cube == reordered_cube


class TestMeshAndTime:
    """Tests for the case where the source cube has two dimensions: mesh and time."""

    def test_order_unchanged(self):
        """Pass a cube with dimensions already in the correct order."""
        cube = mesh_cube()
        cube = add_axis(cube, "time", axis="t")
        original_cube = cube.copy()
        reordered_cube = reorder_cube_dimensions(cube)
        assert original_cube == reordered_cube

    def test_dimensions_reordered(self):
        """Pass a cube with dimensions in the incorrect order."""
        cube = mesh_cube()
        cube = add_axis(cube, "time", axis="t")
        # dimension order is currently as desired: time, mesh
        expected_cube = cube.copy()

        # swap dimensions to the incorrect order: mesh, time
        cube.transpose()
        reordered_cube = reorder_cube_dimensions(cube)

        assert expected_cube == reordered_cube

    def test_dimensions_reordered_with_aux_coord(self):
        """Pass a cube with dimensions in the incorrect order.

        The cube has two time coordinates: one dim and one aux.
        They are both associated with the same dimension.
        """
        cube = mesh_cube()
        cube = add_axis(cube, "time", axis="t")
        new_time_coord = iris.coords.AuxCoord(
            [1, 2], long_name="time_aux", units="days since 1970-01-01 00:00:00"
        )
        cube.add_aux_coord(new_time_coord, 0)
        # dimension order is currently as desired: time, mesh
        expected_cube = cube.copy()

        # swap dimensions to the incorrect order: mesh, time
        cube.transpose()
        reordered_cube = reorder_cube_dimensions(cube)

        assert expected_cube == reordered_cube


class TestMeshAndVertical:
    """Tests for the case where the source cube has two dimensions: mesh and z."""

    def test_order_unchanged(self):
        """Pass a cube with dimensions already in the correct order."""
        cube = four_vertical_levels_cube()
        original_cube = cube.copy()
        reordered_cube = reorder_cube_dimensions(cube)
        assert original_cube == reordered_cube

    def test_dimensions_reordered(self):
        """Pass a cube with dimensions in the incorrect order."""
        cube = four_vertical_levels_cube()
        # Cube has two coords identified as z axis, but only one is a dim coord
        assert len(cube.coords(axis="z")) == 2
        assert len(cube.coords(axis="z", dim_coords=True)) == 1

        # dimension order is currently as desired: model_level_number, mesh
        expected_cube = cube.copy()

        # swap dimensions to the incorrect order: mesh, model_level_number
        cube.transpose()
        reordered_cube = reorder_cube_dimensions(cube)

        assert expected_cube == reordered_cube


@pytest.mark.parametrize("order", tuple(itertools.permutations(range(3))))
class Test3Dimensions:
    """Tests for cases where the source cube has three dimensions.

    Note that these tests are parameterised to transpose the source cube into
    all possible orderings of the three dimensions.
    """

    def test_mesh_vertical_time(self, order):
        """Test when the source cube has three dimensions: Mesh, vertical and time."""
        cube = four_vertical_levels_cube()
        cube = add_axis(cube, "time", axis="t")

        # dimension order is currently as desired: time, model_level_number, mesh
        expected_cube = cube.copy()

        cube.transpose(order)
        reordered_cube = reorder_cube_dimensions(cube)

        assert expected_cube == reordered_cube

    def test_mesh_vertical_other(self, order):
        """Test when the source cube has three dimensions: Mesh, vertical and other."""
        cube = four_vertical_levels_cube()
        cube = add_axis(cube, "other")

        # dimension order is currently as desired: other, model_level_number, mesh
        expected_cube = cube.copy()

        cube.transpose(order)
        reordered_cube = reorder_cube_dimensions(cube)

        assert expected_cube == reordered_cube

    def test_mesh_time_other(self, order):
        """Test when the source cube has three dimensions: Mesh, time and other."""
        cube = mesh_cube()
        cube = add_axis(cube, "time", axis="t")
        cube = add_axis(cube, "other")

        # dimension order is currently as desired: other, time, mesh
        expected_cube = cube.copy()

        cube.transpose(order)
        reordered_cube = reorder_cube_dimensions(cube)

        assert expected_cube == reordered_cube


@pytest.mark.parametrize("order", tuple(itertools.permutations(range(4))))
def test_4_dimensions_reordered(order):
    """Test when the source cube has four dimensions: Mesh, vertical, time and other.

    Note that this test is parameterised to transpose the source cube into
    all possible orderings of the four dimensions.
    """
    cube = four_vertical_levels_cube()
    cube = add_axis(cube, "time", axis="t")
    cube = add_axis(cube, "pseudo_level")

    # dimension order is currently as desired:
    # pseudo_level, time, model_level_number, mesh
    expected_cube = cube.copy()

    # transpose dimensions to incorrect order, then apply reordering function
    cube.transpose(order)
    reordered_cube = reorder_cube_dimensions(cube)

    assert expected_cube == reordered_cube


class TestExceptions:
    """Tests for exceptions raised."""

    def test_multiple_time_dimensions_error(self):
        """Test that an error is raised when multiple time dimensions are present."""
        cube = mesh_cube()
        cube = add_axis(cube, "time", axis="t")
        cube = add_axis(cube, "another_time", axis="t")
        expected_msg = "Multiple time dimensions found: another_time, time."
        with pytest.raises(ValueError, match=expected_msg):
            reorder_cube_dimensions(cube)

    def test_multiple_vertical_dimensions_error(self):
        """Test that an error is raised when multiple vertical dimensions are present."""  # noqa: E501
        cube = mesh_cube()
        cube = add_axis(cube, "vertical", axis="z")
        cube = add_axis(cube, "another_vertical", axis="z")
        expected_msg = "Multiple vertical dimensions found: another_vertical, vertical."
        with pytest.raises(ValueError, match=expected_msg):
            reorder_cube_dimensions(cube)
