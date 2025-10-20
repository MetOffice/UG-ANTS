# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the :class:`ugants.analysis.fill.KDTreeFill` class."""
from unittest import mock

import numpy as np
import pytest
from ugants.analysis.fill import KDTreeFill
from ugants.io import load
from ugants.tests import get_data_path
from ugants.tests.stock import mesh_cube

pytestmark = pytest.mark.filterwarnings(
    "ignore:No cells in the source cube require filling.:UserWarning",
    "ignore:All target mask data is unmasked, target mask has no effect.:UserWarning",
    "ignore:Assuming a spherical geocentric coordinate system for conversion to "
    "3D cartesian coordinates. If the provided cube is not defined on this "
    "coordinate system then unexpected results may occur.:UserWarning",
)


@pytest.fixture()
def sample_data():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")
    return sample_data


@pytest.fixture()
def sample_target_mask(sample_data):
    target_mask = sample_data.copy(np.zeros(sample_data.shape, dtype=int))
    target_mask.data[0] = 1
    return target_mask


class TestCalculateFillLookup:
    @mock.patch("ugants.analysis.fill.KDTree")
    def test_convert_to_cartesian_call(self, patch_kdtree, sample_data):
        # patch the query method of the mocked KDTree
        patch_kdtree.return_value.query.return_value = (None, None)
        with mock.patch("ugants.analysis.fill.convert_to_cartesian") as mock_convert:
            KDTreeFill(sample_data)
        mock_convert.assert_called_once_with(sample_data)

    def test_KDTree_called(self, sample_data):
        with mock.patch("ugants.analysis.fill.KDTree") as mock_kdtree:
            mock_kdtree.return_value.query.return_value = (None, None)
            filler = KDTreeFill(sample_data)
        mock_kdtree.assert_called_once_with(filler.points_to_fill_from)

    @mock.patch("ugants.analysis.fill.KDTree")
    def test_KDTree_queried(self, patch_kdtree, sample_data):
        kdtree_instance = patch_kdtree.return_value
        kdtree_instance.query.return_value = (None, None)
        filler = KDTreeFill(sample_data)
        kdtree_instance.query.assert_called_once_with(filler.points_to_be_filled)


class TestNoTargetMask:
    def test_different_mesh_fail(self, sample_data):
        cube_to_fill = mesh_cube(n_faces=5)
        filler = KDTreeFill(sample_data)
        with pytest.raises(ValueError) as error:
            filler(cube_to_fill)
        assert str(error.value) == "Provided cube and source cube have different meshes"

    def test_nan_cell(self, sample_data):
        source = sample_data.copy()
        source.data[25] = np.nan

        filler = KDTreeFill(source)
        actual = filler(source)

        expected = sample_data.copy()
        expected.data[25] = expected.data[26]

        assert actual == expected

    def test_masked_cell(self):
        data_C4 = load.ugrid(get_data_path("data_C4.nc"))
        sample_data = data_C4.extract_cube("sample_data")

        source = sample_data.copy()
        source.data = np.ma.masked_where(source.data == 26.0, source.data)

        filler = KDTreeFill(source)
        actual = filler(source)

        expected = sample_data.copy()
        expected.data[25] = expected.data[26]

        assert actual == expected

    def test_no_mask_no_nan(self):
        data_C4 = load.ugrid(get_data_path("data_C4.nc"))
        sample_data = data_C4.extract_cube("sample_data")

        source = sample_data.copy()
        filler = KDTreeFill(source)
        actual = filler(source)

        assert actual == sample_data


class TestTargetMask:
    def test_different_mesh_fail(self, sample_data, sample_target_mask):
        cube_to_fill = mesh_cube(n_faces=5)
        filler = KDTreeFill(sample_data, sample_target_mask)
        with pytest.raises(ValueError) as error:
            filler(cube_to_fill)
        assert str(error.value) == "Provided cube and source cube have different meshes"

    def test_one_nan_cell(self, sample_data, sample_target_mask):
        """Fill single missing data point.

        Create source data. One nan cell (X) on first panel of the C4 cube.
        . = regular data
        X = nan value to be filled
        [. . . .]
        [. . X .]
        [. . . .]
        [. . . .]

        Create target mask.
        Mask out all of one panel of the C4 cube, except for two cells.
        1 = masked
        0 = unmasked
        [1 1 1 1]
        [1 0 0 1]
        [1 1 1 1]
        [1 1 1 1]
        """
        sample_data.data[6] = np.nan

        sample_target_mask.data[...] = 1
        sample_target_mask.data[[5, 6]] = 0

        expected = sample_data.copy()
        expected.data[6] = expected.data[5]
        expected.data.mask = sample_target_mask.data

        filler = KDTreeFill(sample_data, sample_target_mask)
        actual = filler(sample_data)

        assert actual == expected

    def test_multiple_nan_cells(self, sample_data, sample_target_mask):
        """Fill multiple missing data points.

        Create source data. Three nan cells (X) on first panel of the C4 cube.
        . = regular data
        X = nan value to be filled
        [. . . .]
        [. . X .]
        [. X X .]
        [. . . .]

        Create target mask.
        Mask out all of one panel of the C4 cube, except for four cells.
        1 = masked
        0 = unmasked
        [1 1 1 1]
        [1 0 0 1]
        [1 0 0 1]
        [1 1 1 1]
        """
        sample_data.data[[6, 9, 10]] = np.nan
        sample_target_mask.data[...] = 1
        sample_target_mask.data[[5, 6, 9, 10]] = 0

        expected = sample_data.copy()
        expected.data[[6, 9, 10]] = expected.data[5]

        filler = KDTreeFill(sample_data, sample_target_mask)
        actual = filler(sample_data)

        assert actual == expected


class TestRepr:
    def test_no_target_mask(self, sample_data):
        filler = KDTreeFill(sample_data)
        expected = (
            "KDTreeFill(source=<iris 'Cube' of sample_data / (1) (-- : 96)>, "
            "target_mask=None)"
        )
        actual = repr(filler)
        assert actual == expected

    def test_with_target_mask(self, sample_data, sample_target_mask):
        filler = KDTreeFill(sample_data, sample_target_mask)
        expected = (
            "KDTreeFill(source=<iris 'Cube' of sample_data / (1) (-- : 96)>, "
            "target_mask=<iris 'Cube' of sample_data / (1) (-- : 96)>)"
        )
        actual = repr(filler)
        assert actual == expected
