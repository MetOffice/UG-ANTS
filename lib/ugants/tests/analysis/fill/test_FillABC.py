# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the :class:`ugants.analysis.fill.FillABC` class."""
from unittest import mock

import numpy as np
import pytest
from iris.cube import Cube
from ugants.analysis.fill import FillABC
from ugants.io import load
from ugants.tests import get_data_path
from ugants.tests.stock import mesh_cube

pytestmark = pytest.mark.filterwarnings(
    "ignore:No cells in the source cube require filling.:UserWarning",
    "ignore:All target mask data is unmasked, target mask has no effect.:UserWarning",
)


@pytest.fixture()
def sample_data():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")
    return sample_data


@pytest.fixture()
def sample_target_mask(sample_data):
    """Return a target mask cube of all zeros (i.e. all unmasked)."""
    target_mask = sample_data.copy(np.zeros(sample_data.shape, dtype=int))
    return target_mask


class DummyConcreteFill(FillABC):
    def calculate_fill_lookup(self):
        pass

    def __call__(self, cube: Cube):
        pass


class TestValidateTargetMask:
    """Tests for the _validate_target_mask method."""

    def test_wrong_dtype_fails(self, sample_data, sample_target_mask):
        """Test that a TypeError is raised when the target mask has the wrong dtype."""
        sample_target_mask.data = sample_target_mask.data.astype(float)
        with pytest.raises(
            TypeError,
            match="Unexpected target mask dtype: float64, expected integer or boolean.",
        ):
            DummyConcreteFill(sample_data, sample_target_mask)

    def test_masked_data_fails(self, sample_data, sample_target_mask):
        """Test that a ValueError is raised when the target mask has masked data."""
        target_mask_mask = np.zeros(sample_target_mask.shape, dtype=bool)
        target_mask_mask[0] = True
        sample_target_mask.data = np.ma.array(
            sample_target_mask.data, mask=target_mask_mask
        )
        assert np.ma.is_masked(sample_target_mask.data)
        with pytest.raises(
            ValueError, match="Unexpected masked data present in target mask."
        ):
            DummyConcreteFill(sample_data, sample_target_mask)

    def test_invalid_mask_value_fails(self, sample_data, sample_target_mask):
        """Test that a ValueError is raised if a value other than 0 or 1 is in target mask."""  # noqa: E501
        sample_target_mask.data[0] = 2
        with pytest.raises(
            ValueError, match="Unexpected values in target mask, got {0, 2}."
        ):
            DummyConcreteFill(sample_data, sample_target_mask)


class TestIdentifyValidFillCells:
    """Tests for the identify_valid_fill_cells method.

    Various combinations of source and target mask are tested by inspecting the
    self.indices_to_fill_from attribute, which is set by the
    identify_valid_fill_cells method in the __init__.
    See the following truth table for expected behaviour for a given cell:

    ================  =====================  ====================
    Masked in source  Masked in target_mask  Valid fill candidate
    ================  =====================  ====================
    False             False                  True
    False             True                   False
    True              False                  False
    True              True                   False
    ================  =====================  ====================
    """

    def test_all_valid_source_no_target_mask(self, sample_data):
        """All source data is unmasked."""
        filler = DummyConcreteFill(sample_data)
        expected = np.full(sample_data.shape, True)
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_one_invalid_source_no_target_mask(self, sample_data):
        """One source data cell is masked."""
        sample_data.data[0] = np.ma.masked
        filler = DummyConcreteFill(sample_data)
        expected = np.full(sample_data.shape, True)
        expected[0] = False
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_all_valid_source_and_target_mask(self, sample_data, sample_target_mask):
        """All source data is unmasked. All target mask data is unmasked."""
        filler = DummyConcreteFill(sample_data, sample_target_mask)
        expected = np.full(sample_data.shape, True)
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_all_valid_source_one_invalid_target_mask(
        self, sample_data, sample_target_mask
    ):
        """All source data is unmasked. One cell in target mask is masked."""
        sample_target_mask.data[0] = 1
        with mock.patch("warnings.warn") as warn:
            filler = DummyConcreteFill(sample_data, sample_target_mask)
        warn.assert_called_once_with(
            "No cells in the source cube require filling.", stacklevel=1
        )

        expected = np.full(sample_data.shape, True)
        expected[0] = False
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_one_invalid_source_all_valid_target_mask(
        self, sample_data, sample_target_mask
    ):
        """One cell in the source is masked, no cells in the target mask are masked."""
        sample_data.data[0] = np.ma.masked
        with mock.patch("warnings.warn") as warn:
            filler = DummyConcreteFill(sample_data, sample_target_mask)
        warn.assert_called_once_with(
            "All target mask data is unmasked, target mask has no effect.", stacklevel=1
        )

        expected = np.full(sample_data.shape, True)
        expected[0] = False
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_one_invalid_source_one_invalid_target_mask(
        self, sample_data, sample_target_mask
    ):
        """One cell in the source is masked, one cell in the target mask is masked.

        They are in different locations.
        """
        sample_data.data[0] = np.ma.masked
        sample_target_mask.data[1] = 1
        filler = DummyConcreteFill(sample_data, sample_target_mask)
        expected = np.full(sample_data.shape, True)
        expected[[0, 1]] = False
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_same_cell_masked_in_source_and_target_mask(
        self, sample_data, sample_target_mask
    ):
        """One cell in the source is masked, one cell in the target mask is masked.

        They are in the same location.
        """
        sample_data.data[0] = np.ma.masked
        sample_target_mask.data[0] = 1
        filler = DummyConcreteFill(sample_data, sample_target_mask)
        expected = np.full(sample_data.shape, True)
        expected[0] = False
        actual = filler.indices_to_fill_from
        np.testing.assert_array_equal(actual, expected)

    def test_no_valid_fill_candidates(self, sample_data):
        """If no valid fill candidates are found, a ValueError should be raised."""
        sample_data.data = np.full(sample_data.shape, np.nan)
        with pytest.raises(
            ValueError, match="No valid fill candidates in source cube."
        ):
            DummyConcreteFill(sample_data)


class TestIdentifyCellsToFill:
    """Tests for the identify_cells_to_fill method.

    Various combinations of source and target mask are tested by inspecting the
    self.indices_to_be_filled attribute, which is set by the
    identify_cells_to_fill method in the __init__.
    See the following truth table for expected behaviour for a given cell:

    ================  =====================  =====================
    Masked in source  Masked in target_mask  Cell requires filling
    ================  =====================  =====================
    False             False                  False
    False             True                   False
    True              False                  True
    True              True                   False
    ================  =====================  =====================
    """

    def test_all_unmasked_source_no_target_mask(self, sample_data):
        """All source data is unmasked."""
        with mock.patch("warnings.warn") as warn:
            filler = DummyConcreteFill(sample_data)
        warn.assert_called_once_with(
            "No cells in the source cube require filling.", stacklevel=1
        )
        expected = np.full(sample_data.shape, False)
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)

    def test_one_masked_source_no_target_mask(self, sample_data):
        """One source data cell is masked."""
        sample_data.data[0] = np.ma.masked
        filler = DummyConcreteFill(sample_data)
        expected = np.full(sample_data.shape, False)
        expected[0] = True
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)

    def test_all_unmasked_source_and_target_mask(self, sample_data, sample_target_mask):
        """All source data is unmasked. All target mask data is unmasked."""
        with mock.patch("warnings.warn") as warn:
            filler = DummyConcreteFill(sample_data, sample_target_mask)
        warn.assert_has_calls(
            [
                mock.call("No cells in the source cube require filling.", stacklevel=1),
                mock.call(
                    "All target mask data is unmasked, target mask has no effect.",
                    stacklevel=1,
                ),
            ]
        )
        expected = np.full(sample_data.shape, False)
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)

    def test_all_unmasked_source_one_masked_target_mask(
        self, sample_data, sample_target_mask
    ):
        """All source data is unmasked. One cell in target mask is masked."""
        sample_target_mask.data[0] = 1
        with mock.patch("warnings.warn") as warn:
            filler = DummyConcreteFill(sample_data, sample_target_mask)
        warn.assert_called_once_with(
            "No cells in the source cube require filling.", stacklevel=1
        )
        expected = np.full(sample_data.shape, False)
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)

    def test_one_masked_source_all_unmasked_target_mask(
        self, sample_data, sample_target_mask
    ):
        """One cell in the source is masked, no cells in the target mask are masked."""
        sample_data.data[0] = np.ma.masked
        with mock.patch("warnings.warn") as warn:
            filler = DummyConcreteFill(sample_data, sample_target_mask)
        warn.assert_called_once_with(
            "All target mask data is unmasked, target mask has no effect.", stacklevel=1
        )
        expected = np.full(sample_data.shape, False)
        expected[0] = True
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)

    def test_one_masked_source_one_masked_target_mask(
        self, sample_data, sample_target_mask
    ):
        """One cell in the source is masked, one cell in the target mask is masked.

        They are in different locations.
        """
        sample_data.data[0] = np.ma.masked
        sample_target_mask.data[1] = 1
        filler = DummyConcreteFill(sample_data, sample_target_mask)
        expected = np.full(sample_data.shape, False)
        expected[0] = True
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)

    def test_same_cell_masked_in_source_and_target_mask(
        self, sample_data, sample_target_mask
    ):
        """One cell in the source is masked, one cell in the target mask is masked.

        They are in the same location.
        """
        sample_data.data[0] = np.ma.masked
        sample_target_mask.data[0] = 1
        filler = DummyConcreteFill(sample_data, sample_target_mask)
        expected = np.full(sample_data.shape, False)
        actual = filler.indices_to_be_filled
        np.testing.assert_array_equal(actual, expected)


class TestInit:
    def test_repr_no_mask(self, sample_data):
        dummy_instance = DummyConcreteFill(sample_data)
        expected = (
            "DummyConcreteFill(source=<iris 'Cube' of sample_data / (1) (-- : 96)>, "
            "target_mask=None)"
        )
        actual = repr(dummy_instance)
        assert actual == expected

    def test_repr_with_mask(self, sample_data, sample_target_mask):
        dummy_instance = DummyConcreteFill(sample_data, sample_target_mask)
        expected = (
            "DummyConcreteFill(source=<iris 'Cube' of sample_data / (1) (-- : 96)>, "
            "target_mask=<iris 'Cube' of sample_data / (1) (-- : 96)>)"
        )
        actual = repr(dummy_instance)
        assert actual == expected

    def test_different_target_mesh_fail(self, sample_data):
        target_mask = mesh_cube(n_faces=5)
        with pytest.raises(ValueError) as error:
            DummyConcreteFill(sample_data, target_mask)
        assert str(error.value) == "Source and target mask have different meshes"

    def test_convert_nan_to_masked_call(self, sample_data):
        convert_nan_to_masked_target = "ugants.analysis.fill.convert_nan_to_masked"
        with mock.patch(convert_nan_to_masked_target) as mock_convert:
            mock_convert.return_value = sample_data
            DummyConcreteFill(sample_data)
        mock_convert.assert_called_once_with(sample_data)

    def test_identify_cells_to_fill_call(self, sample_data):
        with mock.patch(
            "ugants.analysis.fill.FillABC.identify_cells_to_fill"
        ) as mock_method:
            # Because we are mocking out the identify_cells_to_fill method,
            # the attribute self.indices_to_be_filled never gets set. This will
            # result in an AttributeError when trying to use self.indices_to_be_filled
            # to set self.points_to_be_filled later on in the __init__
            try:
                DummyConcreteFill(sample_data)
            except AttributeError:
                pass
        mock_method.assert_called_once_with()

    def test_identify_valid_fill_cells_call(self, sample_data):
        with mock.patch(
            "ugants.analysis.fill.FillABC.identify_valid_fill_cells"
        ) as mock_method:
            # Because we are mocking out the identify_valid_fill_cells method,
            # the attribute self.indices_to_fill_from never gets set. This will
            # result in an AttributeError when trying to use self.indices_to_fill_from
            # to set self.points_to_fill_from later on in the __init__
            try:
                DummyConcreteFill(sample_data)
            except AttributeError:
                pass
        mock_method.assert_called_once_with()
