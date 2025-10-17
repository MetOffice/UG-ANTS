# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import pytest

from ugants.regrid.band_utils import (
    generate_band_bounds,
)


class TestGenerateBandsBounds:
    def test_correct_number_of_bands(self):
        bands = generate_band_bounds(-10, 10, 5)
        assert len(bands) == 5

    def test_correct_values_of_bands(self):
        expected_bounds_values = [
            (-10.0, -6.0),
            (-6.0, -2.0),
            (-2.0, +2.0),
            (+2.0, +6.0),
            (+6.0, +10.0),
        ]
        bands = generate_band_bounds(-10, 10, 5)
        assert bands == expected_bounds_values

    def test_n_bands_less_than_two(self):
        with pytest.raises(ValueError, match="n_bands must be 2 or more. Only got 1"):
            generate_band_bounds(-10, 10, 1)
