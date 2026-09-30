# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest
from ugants.tests.stock import cubedsphere_cube
from ugants.utils.cube import Stencil


@pytest.fixture()
def source():
    return cubedsphere_cube(4)


@pytest.mark.parametrize("central_cell_index", [5, -91])
class TestMidPanel:
    #                 +---+---+---+---+
    #                 |35 |39 |43 |47 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |34 |38 |42 |46 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |33 |37 |41 |45 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |32 |36 |40 |44 |
    #                 | 3 | 3 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |64 |65 |66 |67 |0  |1  |2  |3  |
    # | 0 | 0 | 0 | 3 | 2 | 1 | 2 | 3 |
    # +---+---+---+---+---+---+---+---+
    # |68 |69 |70 |71 |4  |5  |6  |7  |
    # | 0 | 0 | 0 | 3 | 1 | 0 | 1 | 3 |
    # +---+---+---+---+---+---+---+---+
    # |72 |73 |74 |75 |8  |9  |10 |11 |
    # | 0 | 0 | 0 | 3 | 2 | 1 | 2 | 3 |
    # +---+---+---+---+---+---+---+---+
    # |76 |77 |78 |79 |12 |13 |14 |15 |
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
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 32, 36, 40, 67, 71, 75]
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
    #                 |35 |39 |43 |47 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |34 |38 |42 |46 |
    #                 | 0 | 0 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |33 |37 |41 |45 |
    #                 | 3 | 3 | 0 | 0 |
    #                 +---+---+---+---+
    #                 |32 |36 |40 |44 |
    #                 | 1 | 2 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |64 |65 |66 |67 |0  |1  |2  |3  |
    # | 0 | 0 | 3 | 1 | 0 | 1 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |68 |69 |70 |71 |4  |5  |6  |7  |
    # | 0 | 0 | 3 | 2 | 1 | 2 | 3 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |72 |73 |74 |75 |8  |9  |10 |11 |
    # | 0 | 0 | 0 | 3 | 3 | 3 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    # |76 |77 |78 |79 |12 |13 |14 |15 |
    # | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    # +---+---+---+---+---+---+---+---+
    def test_immediate_neighbours(self, central_cell_index, source):
        expected = np.array([0, 1, 4, 32, 67])
        expected.sort()
        stencil = Stencil(source, iterations=1)
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)

    def test_extended_neighbours(self, central_cell_index, source):
        expected = np.array([0, 1, 4, 5, 32, 36, 67, 71])
        expected.sort()
        stencil = Stencil(source, iterations=2)
        central_cell_index = 0
        actual = stencil[central_cell_index]
        actual.sort()
        np.testing.assert_array_equal(actual, expected)

    def test_third_iteration(self, central_cell_index, source):
        expected = np.array(
            [0, 1, 4, 5, 32, 36, 67, 71, 33, 37, 40, 66, 2, 70, 6, 75, 8, 9]
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
