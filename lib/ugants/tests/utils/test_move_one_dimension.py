# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Tests for the dimension-moving utility function.

In addition to :func:`ugants.utils.move_one_dimension`, its private support functions
are also tested separately.
"""
from unittest import mock

import numpy as np
import pytest

from ugants.utils import (
    _check_and_normalise_index_in_range,
    _one_dimension_transpose_indices,
    move_one_dimension,
)


class TestIndexChecking:
    """Test the function which produces dimension permuting indices."""

    def test_basic(self):
        result = _check_and_normalise_index_in_range(2, 3, "myargname")
        expected = 2
        assert result == expected

    def test_zerodim_0(self):
        assert _check_and_normalise_index_in_range(0, 1, "xxx") == 0

    def test_onedimensional_0(self):
        assert _check_and_normalise_index_in_range(0, 1, "xxx") == 0

    def test_onedimensional_minus1(self):
        assert _check_and_normalise_index_in_range(-1, 1, "xxx") == 0

    def test_onedimensional_1__fails(self):
        expected = (
            "ValueError("
            "\"Value of 'xxx' arg is out of valid range -1 to 0 : got 1.\""
            ")"
        )
        with pytest.raises(Exception) as error:
            _check_and_normalise_index_in_range(1, 1, "xxx")
        assert repr(error.value) == expected

    def test_onedimensional_minus2__fails(self):
        expected = (
            "ValueError("
            "\"Value of 'xxx' arg is out of valid range -1 to 0 : got -2.\""
            ")"
        )
        with pytest.raises(Exception) as error:
            _check_and_normalise_index_in_range(-2, 1, "xxx")
        assert repr(error.value) == expected

    def test_multidim_positive_inner_index(self):
        assert _check_and_normalise_index_in_range(4, 7, "xxx") == 4

    def test_multidim_negative_inner_index(self):
        """N.B. this test also verifies the negative index conversion behaviour."""
        assert _check_and_normalise_index_in_range(-4, 7, "xxx") == 3

    def test_multidim_biggest_index(self):
        assert _check_and_normalise_index_in_range(6, 7, "xxx") == 6

    def test_multidim_toobig_index(self):
        expected = (
            "ValueError("
            "\"Value of 'xxx' arg is out of valid range -7 to 6 : got 7.\""
            ")"
        )
        with pytest.raises(Exception) as error:
            _check_and_normalise_index_in_range(7, 7, "xxx")
        assert repr(error.value) == expected

    def test_multidim_smallest_negative_index(self):
        assert _check_and_normalise_index_in_range(-7, 7, "xxx") == 0

    def test_multidim_toosmall_negative_index(self):
        expected = (
            "ValueError("
            "\"Value of 'xxx' arg is out of valid range -7 to 6 : got -8.\""
            ")"
        )
        with pytest.raises(Exception) as error:
            _check_and_normalise_index_in_range(-8, 7, "xxx")
        assert repr(error.value) == expected


class TestPrivateFunctionTransposeIndices:
    """
    Test the subsidiary function which produces dimension permuting indices.

    "_one_dimension_transpose_indices" is effectively a private internal utll, only
    there to support "move_one_dimension".
    """

    def test_basic(self):
        """Exercise a simple concrete test-case."""
        result = _one_dimension_transpose_indices(
            number_of_dimensions=5, from_dim=1, to_dim=3
        )
        expected = [0, 2, 3, 1, 4]
        assert result == expected

    def test_concrete_negative_indices(self):
        """Another simple testcase, this time using negative indices."""
        result = _one_dimension_transpose_indices(
            number_of_dimensions=6, from_dim=-3, to_dim=-5
        )
        expected = [0, 3, 1, 2, 4, 5]
        assert result == expected

    def test_calls_validation_function(self):
        """Check that the transpose function calls the validation function."""
        target = "ugants.utils._check_and_normalise_index_in_range"
        mock_check_function = mock.Mock(side_effect=[4, 7])
        with mock.patch(target, mock_check_function):
            _one_dimension_transpose_indices(17, 71, 72)
        assert mock_check_function.call_args_list == [
            mock.call(71, 17, "from_dim"),
            mock.call(72, 17, "to_dim"),
        ]

    # NOTE: given the above, we don't provide any other tests for bad index values.

    def test_zerodimensions__fail(self):
        with pytest.raises(Exception) as error:
            _one_dimension_transpose_indices(0, 0, 0)
        assert repr(error.value) == "ValueError('Cannot transpose with 0 dimensions.')"

    def test_onedimension_0_to_0__nochange(self):
        assert _one_dimension_transpose_indices(1, 0, 0) == [0]

    def test_multidimensional_inner_index__nochange(self):
        assert _one_dimension_transpose_indices(6, 3, 3) == [0, 1, 2, 3, 4, 5]

    def test_multidimensional_inner_to_max_positive(self):
        assert _one_dimension_transpose_indices(6, 3, 5) == [0, 1, 2, 4, 5, 3]

    def test_multidimensional_inner_to_minus1(self):
        """Same result as previous."""
        assert _one_dimension_transpose_indices(6, 3, -1) == [0, 1, 2, 4, 5, 3]

    def test_multidimensional_inner_to_zero(self):
        assert _one_dimension_transpose_indices(6, 3, 0) == [3, 0, 1, 2, 4, 5]

    def test_multidimensional_zero_to_inner(self):
        assert _one_dimension_transpose_indices(6, 0, 2) == [1, 2, 0, 3, 4, 5]

    def test_multidimensional_max_to_inner(self):
        assert _one_dimension_transpose_indices(6, 5, 2) == [0, 1, 5, 2, 3, 4]

    def test_multidimensional_minus2_to_inner(self):
        assert _one_dimension_transpose_indices(6, -2, 2) == [0, 1, 4, 2, 3, 5]


class TestCall:
    def test_zero_dims(self):
        """Check what happens if the passed object has 0 dims."""
        testobject = np.array(1.0)  # a scalar array object
        with pytest.raises(ValueError, match="Cannot transpose with 0 dimensions."):
            move_one_dimension(testobject, 0, 0)

    def test_simple(self):
        """Check a specific, concrete testcase."""
        test_dims = [3, 2, 4, 6, 5]
        test_from_index = 1
        test_to_index = 3
        test_array = np.arange(np.prod(test_dims)).reshape(test_dims)
        expected = test_array.transpose([0, 2, 3, 1, 4])

        result = move_one_dimension(
            test_array, from_dim=test_from_index, to_dim=test_to_index
        )

        # The result should match the "expected" one from the equivalent transpose call.
        assert np.all(result == expected)

    @pytest.fixture()
    def mocked_indices_call(self):
        target = "ugants.utils._one_dimension_transpose_indices"
        with mock.patch(target) as mock_indices_call:
            yield mock_indices_call

    def test_calls_indices_function(self, mocked_indices_call):
        """
        Check that it calls the related function.

        See that it calls '_one_dimension_transpose_indices' with the expected args.
        """
        mock_arraylike = mock.Mock()
        mock_fromindex = mock.sentinel.fromindex
        mock_toindex = mock.sentinel.toindex
        move_one_dimension(mock_arraylike, mock_fromindex, mock_toindex)
        mocked_indices_call.assert_called_once_with(
            mock_arraylike.ndim, mock_fromindex, mock_toindex
        )

    def test_calls_argument_transpose(self, mocked_indices_call):
        """Check the call to the ".transpose()" method of the argument."""
        mock_arraylike = mock.Mock()
        mock_fromindex = mock.sentinel.fromindex
        mock_toindex = mock.sentinel.toindex

        # Exercise the actual call.
        result = move_one_dimension(mock_arraylike, mock_fromindex, mock_toindex)

        # Check the transpose call args.
        transpose_method = mock_arraylike.transpose
        transpose_method.assert_called_once_with(mocked_indices_call.return_value)
        # Check that the transpose result was the final returned value.
        assert result == transpose_method.return_value
