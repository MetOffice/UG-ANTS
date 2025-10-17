# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest
from ugants.io.load import ugrid
from ugants.tests import get_data_path
from ugants.utils.cube import Stencil


@pytest.fixture()
def source():
    return ugrid(get_data_path("data_C4.nc")).extract_cube("sample_data")


@pytest.mark.parametrize("central_cell_index", [5, -91])
class TestMidPanel:
    #                 +---+---+---+---+
    #                 |67 |71 |75 |79 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |66 |70 |74 |78 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |65 |69 |73 |77 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |64 |68 |72 |76 |
    #                 | 3 | 3 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |48 |49 |50 |51 |0  |1  |2  |3  |
    # | 0 | 0 | 0 | 3 | 2 | 1 | 2 | 3 |
    # +---+---+---+---+---+---+---+---+
    # |52 |53 |54 |55 |4  |5  |6  |7  |
    # | 0 | 0 | 0 | 3 | 1 | 0 | 1 | 3 |
    # +---+---+---+---+---+---+---+---+
    # |56 |57 |58 |59 |8  |9  |10 |11 |
    # | 0 | 0 | 0 | 3 | 2 | 1 | 2 | 3 |
    # +---+---+---+---+---+---+---+---+
    # |60 |61 |62 |63 |12 |13 |14 |15 |
    # | 0 | 0 | 0 | 0 | 3 | 3 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    def test_immediate_neigbours(self, central_cell_index, source):
        expected = np.array([1, 4, 5, 6, 9])
        stencil = Stencil(source, iterations=1)
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)

    def test_extended_neighbours(self, central_cell_index, source):
        expected = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
        stencil = Stencil(source, iterations=2)
        central_cell_index = 5
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)

    def test_third_iteration(self, central_cell_index, source):
        expected = np.array(
            [64, 68, 72, 51, 0, 1, 2, 3, 55, 4, 5, 6, 7, 59, 8, 9, 10, 11, 12, 13, 14]
        )
        expected.sort()
        stencil = Stencil(source, iterations=3)
        central_cell_index = 5
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("central_cell_index", [0, -96])
class TestPanelCorner:
    #                 +---+---+---+---+
    #                 |67 |71 |75 |79 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |66 |70 |74 |78 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |65 |69 |73 |77 |
    #                 | 3 | 3 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |64 |68 |72 |76 |
    #                 | 1 | 2 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |48 |49 |50 |51 |0  |1  |2  |3  |
    # | 0 | 0 | 3 | 1 | 0 | 1 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |52 |53 |54 |55 |4  |5  |6  |7  |
    # | 0 | 0 | 3 | 2 | 1 | 2 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |56 |57 |58 |59 |8  |9  |10 |11 |
    # | 0 | 0 | 0 | 3 | 3 | 3 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |60 |61 |62 |63 |12 |13 |14 |15 |
    # | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    def test_immediate_neighbours(self, central_cell_index, source):
        expected = np.array([64, 51, 0, 1, 4])
        expected.sort()
        stencil = Stencil(source, iterations=1)
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)

    def test_extended_neighbours(self, central_cell_index, source):
        expected = np.array([64, 68, 51, 0, 1, 55, 4, 5])
        expected.sort()
        stencil = Stencil(source, iterations=2)
        central_cell_index = 0
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)

    def test_third_iteration(self, central_cell_index, source):
        expected = np.array(
            [65, 69, 64, 68, 72, 50, 51, 0, 1, 2, 54, 55, 4, 5, 6, 59, 8, 9]
        )
        expected.sort()
        stencil = Stencil(source, iterations=3)
        central_cell_index = 0
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)


class TestIndexErrors:
    def test_index_over_range(self, source):
        stencil = Stencil(source)
        with pytest.raises(
            IndexError,
            match="Cannot index face 96 for array of length 96.",
        ):
            stencil[96]

    def test_index_under_range(self, source):
        stencil = Stencil(source)
        with pytest.raises(
            IndexError,
            match="Cannot index face -97 for array of length 96.",
        ):
            stencil[-97]


class TestInvalidIterationErrors:
    def test_non_integer(self, source):
        with pytest.raises(
            TypeError,
            match="iterations must be an int, got <class 'str'>.",
        ):
            Stencil(source, "iterations")

    @pytest.mark.parametrize("iterations", [0, -1])
    def test_non_positive_integer(self, iterations, source):
        with pytest.raises(
            ValueError,
            match=f"iterations must be a positive integer, got {iterations}.",
        ):
            Stencil(source, iterations)
