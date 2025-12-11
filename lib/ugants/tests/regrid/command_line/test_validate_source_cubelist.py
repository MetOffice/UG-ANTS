# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris
import pytest
from iris.cube import CubeList

from ugants.regrid.command_line import (
    _validate_source_cubelist_length,
    _validate_source_is_global,
)
from ugants.tests.stock import regular_grid_global_cube


@pytest.fixture()
def regular_grid_global_cube_cubelist():
    """Return a single-element CubeList of regular lat-lon data.

    The cube's latitude and longitude coordinates have no bounds.
    """
    cube = regular_grid_global_cube(144, 192)
    cube.coord(axis="x").bounds = None
    cube.coord(axis="y").bounds = None
    return CubeList([cube])


class TestValidateSourceCubeListLength:
    def test_more_than_one_source_fails(self, regular_grid_global_cube_cubelist):
        """Test that passing a two element CubeList fails."""
        invalid_source = CubeList(regular_grid_global_cube_cubelist * 2)
        assert len(invalid_source) == 2
        with pytest.raises(ValueError, match="Source contained 2 cubes, expected 1."):
            _validate_source_cubelist_length(invalid_source)


class TestValidateSourceIsGlobal:
    def test_non_global_latitude_no_bounds(self, regular_grid_global_cube_cubelist):
        """Test that if the source does not cover +/-90 latitude, an error is raised.

        In this case the latitude coordinate does not have bounds.
        """
        invalid_source = regular_grid_global_cube_cubelist.extract_cube(
            iris.Constraint(latitude=lambda cell: cell < 80)
        )
        latitude = invalid_source.coord("latitude").copy()
        latitude.guess_bounds()
        assert latitude.bounds.max() == 80

        assert not invalid_source.coord("latitude").has_bounds()
        with pytest.raises(
            ValueError,
            match="The provided source is not global: latitude min = -90.0, "
            "latitude max = 80.0",
        ):
            _validate_source_is_global(invalid_source)

    def test_non_global_latitude_with_bounds(self, regular_grid_global_cube_cubelist):
        """Test that if the source does not cover +/-90 latitude, an error is raised.

        In this case the latitude coordinate has bounds.
        """
        invalid_source = regular_grid_global_cube_cubelist.extract_cube(
            iris.Constraint(latitude=lambda cell: cell < 80)
        )
        invalid_source.coord("latitude").guess_bounds()
        assert invalid_source.coord("latitude").bounds.max() == 80

        with pytest.raises(
            ValueError,
            match="The provided source is not global: latitude min = -90.0, "
            "latitude max = 80.0",
        ):
            _validate_source_is_global(invalid_source)

    def test_global_latitude_no_bounds(self, regular_grid_global_cube_cubelist):
        """Test that no error is raised when the source covers full latitude.

        In this case the latitude coordinate does not have bounds.
        """
        valid_source = regular_grid_global_cube_cubelist.extract_cube("sample_data")
        latitude = valid_source.coord("latitude").copy()
        latitude.guess_bounds()
        assert latitude.bounds.max() == 90
        assert latitude.bounds.min() == -90

        assert not valid_source.coord("latitude").has_bounds()
        _validate_source_is_global(valid_source)

    def test_global_latitude_with_bounds(self, regular_grid_global_cube_cubelist):
        """Test that no error is raised when the source covers full latitude.

        In this case the latitude coordinate has bounds.
        """
        valid_source = regular_grid_global_cube_cubelist.extract_cube("sample_data")
        valid_source.coord("latitude").guess_bounds()
        assert valid_source.coord("latitude").bounds.max() == 90
        assert valid_source.coord("latitude").bounds.min() == -90

        _validate_source_is_global(valid_source)

    def test_non_global_longitude(self, regular_grid_global_cube_cubelist):
        """Test if the source does not cover 360 degrees longitude, an error is raised."""  # noqa: E501
        invalid_source = regular_grid_global_cube_cubelist.extract_cube(
            iris.Constraint(longitude=lambda cell: cell < 80)
        )
        assert not invalid_source.coord("longitude").circular
        with pytest.raises(
            ValueError,
            match="The provided source is not global: longitude is not circular.",
        ):
            _validate_source_is_global(invalid_source)
