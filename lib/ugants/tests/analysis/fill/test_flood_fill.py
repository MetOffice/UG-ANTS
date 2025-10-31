# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the :func:`ugants.analysis.fill.flood_fill function."""

import pytest
import ugants.tests
from iris.coords import DimCoord
from iris.cube import CubeList
from ugants.analysis.fill import flood_fill
from ugants.io import load


@pytest.fixture()
def source():
    source = load.ugrid(ugants.tests.get_data_path("data_C4.nc")).extract_cube(
        "sample_data"
    )
    return source


def test_multiple_dimensions_error(source):
    """Test that an error is raised if a multi dimensional cube is passed."""
    # Create a cube with a vertical dimension
    vertical_levels = DimCoord([0, 1], "height")
    second_level = source.copy()
    source.add_aux_coord(vertical_levels[0])
    second_level.add_aux_coord(vertical_levels[1])
    multi_dim_cube = CubeList([source, second_level]).merge()[0]

    with pytest.raises(
        ValueError,
        match="The provided cube has 2 data dimensions, "
        "expected only 1 horizontal dimension.",
    ):
        flood_fill(multi_dim_cube, 0, 0)


def test_already_filled_error(source):
    """Test that an error is raised when trying to fill from an already filled point."""
    seed_point = 5
    fill_value = 2
    source.data[seed_point] = fill_value
    with pytest.raises(
        ValueError, match="The value at location 5 already has the fill value 2."
    ):
        flood_fill(source, seed_point, fill_value)


def test_no_inplace_operation(source):
    """Test that the source cube is not modified in place."""
    # Set a few adjacent points to the same value
    # so that a flood fill does occur.
    source.data[:3] = 0

    original_source = source.copy()
    actual = flood_fill(source, 0, 1)

    # Check flood fill has taken place
    assert actual != source
    # And that the source has not been modified
    assert original_source == source


def test_single_panel_fill(source):
    """Test flood filling constrained to a single panel."""
    # -= Source cube =-
    # Seed point is cell 5, marked with an X in the upper right corner.
    # Key:
    # +---+
    # |IN |<-- Cell index
    # | D |<-- Cell data value pre-fill
    # +---+
    #
    # +---+---+---+---+
    # |0  |1  |2  |3  |
    # | 1 | 0 | 1 | 0 |
    # +---+---+---+---+
    # |4  |5 X|6  |7  |
    # | 0 | 1 | 1 | 1 |
    # +---+---+---+---+
    # |8  |9  |10 |11 |
    # | 0 | 1 | 0 | 1 |
    # +---+---+---+---+
    # |12 |13 |14 |15 |
    # | 1 | 0 | 0 | 0 |
    # +---+---+---+---+
    #
    # -= Filled cube =-
    # Fill value is 2.
    # Key:
    # +---+
    # |IN |<-- Cell index
    # | D |<-- Cell data value pre-fill
    # +---+
    #
    # +---+---+---+---+
    # |0  |1  |2  |3  |
    # | 1 | 0 | 2 | 0 |
    # +---+---+---+---+
    # |4  |5 X|6  |7  |
    # | 0 | 2 | 2 | 2 |
    # +---+---+---+---+
    # |8  |9  |10 |11 |
    # | 0 | 2 | 0 | 2 |
    # +---+---+---+---+
    # |12 |13 |14 |15 |
    # | 1 | 0 | 0 | 0 |
    # +---+---+---+---+

    # Set all cells to 0
    source.data[:] = 0

    # Set the points on the first panel following the
    # pattern outlined in diagrams above
    source.data[[0, 2, 5, 6, 7, 9, 11, 12]] = 1

    fill_value = 2
    expected_filled = source.copy()
    expected_filled.data[[2, 5, 6, 7, 9, 11]] = fill_value

    actual_filled = flood_fill(source, 5, fill_value)
    assert actual_filled == expected_filled


def test_multi_panel_fill(source):
    """Test flood fill over a panel corner."""
    # -= Source cube =-
    # Seed point is cell 0, marked with an X in the upper right corner.
    # Key:
    # +---+
    # |IN |<-- Cell index
    # | D |<-- Cell data value pre-fill
    # +---+
    #                 +---+---+---+---+
    #                 |67 |71 |75 |79 |
    #                 | 0 | 0 | 0 | 1 |
    #                 +---+---+---+---+
    #                 |66 |70 |74 |78 |
    #                 | 0 | 0 | 0 | 1 |
    #                 +---+---+---+---+
    #                 |65 |69 |73 |77 |
    #                 | 1 | 1 | 1 | 0 |
    #                 +---+---+---+---+
    #                 |64 |68 |72 |76 |
    #                 | 0 | 0 | 0 | 1 |
    # +---+---+---+---+---+---+---+---+
    # |48 |49 |50 |51 |0 X|1  |2  |3  |
    # | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
    # +---+---+---+---+---+---+---+---+
    # |52 |53 |54 |55 |4  |5  |6  |7  |
    # | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |56 |57 |58 |59 |8  |9  |10 |11 |
    # | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |60 |61 |62 |63 |12 |13 |14 |15 |
    # | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    #
    # -= Filled cube =-
    # Fill value is 2.
    # Key:
    # +---+
    # |IN |<-- Cell index
    # | D |<-- Cell data value post-fill
    # +---+
    #                 +---+---+---+---+
    #                 |67 |71 |75 |79 |
    #                 | 0 | 0 | 0 | 1 |
    #                 +---+---+---+---+
    #                 |66 |70 |74 |78 |
    #                 | 0 | 0 | 0 | 1 |
    #                 +---+---+---+---+
    #                 |65 |69 |73 |77 |
    #                 | 2 | 2 | 2 | 0 |
    #                 +---+---+---+---+
    #                 |64 |68 |72 |76 |
    #                 | 0 | 0 | 0 | 2 |
    # +---+---+---+---+---+---+---+---+
    # |48 |49 |50 |51 |0 X|1  |2  |3  |
    # | 1 | 0 | 2 | 2 | 2 | 2 | 2 | 2 |
    # +---+---+---+---+---+---+---+---+
    # |52 |53 |54 |55 |4  |5  |6  |7  |
    # | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |56 |57 |58 |59 |8  |9  |10 |11 |
    # | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |60 |61 |62 |63 |12 |13 |14 |15 |
    # | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+

    # Set all cells to 0
    source.data[:] = 0

    # Set the points following the
    # pattern outlined in diagrams above
    source.data[[0, 1, 2, 3, 5, 48, 50, 51, 65, 69, 73, 76, 78, 79]] = 1

    fill_value = 2
    expected_filled = source.copy()
    expected_filled.data[[0, 1, 2, 3, 5, 50, 51, 65, 69, 73, 76]] = fill_value

    actual_filled = flood_fill(source, 5, fill_value)
    assert actual_filled == expected_filled
