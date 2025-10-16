# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import pytest

from ugants.regrid.band_utils import (
    _add_padding,
)


class TestAddPadding:
    def test_default_padding(self):
        expected = (9.0, 21.0)

        actual = _add_padding(10.0, 20.0)

        assert actual == expected

    def test_non_default_padding(self):
        expected = (-22.0, -8.0)

        actual = _add_padding(-20.0, -10.0, padding_fraction=0.2)

        assert actual == expected

    def test_error_min_greater_than_max(self):
        with pytest.raises(
            ValueError, match="Minimum bound 1.0 is not below maximum bound 0.0."
        ):
            _add_padding(1.0, 0.0)

    def test_error_min_equal_to_max(self):
        with pytest.raises(
            ValueError, match="Minimum bound 0.0 is not below maximum bound 0.0."
        ):
            _add_padding(0.0, 0.0)
