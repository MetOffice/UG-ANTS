# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import numpy as np
from ugants.analysis.fill import convert_nan_to_masked
from ugants.tests.stock import cubedsphere_cube


def test_no_nan_no_mask():
    sample_data = cubedsphere_cube(4)

    actual = convert_nan_to_masked(sample_data)

    assert actual == sample_data


def test_no_nan_one_masked_cell():
    sample_data = cubedsphere_cube(4)
    sample_data.data[0] = np.ma.masked

    actual = convert_nan_to_masked(sample_data)

    assert actual == sample_data


def test_one_nan_cell_no_mask():
    sample_data = cubedsphere_cube(4)
    sample_data.data[0] = np.nan

    expected = sample_data.copy()
    expected.data[0] = np.ma.masked

    actual = convert_nan_to_masked(sample_data)

    assert actual == expected


def test_one_nan_one_masked():
    sample_data = cubedsphere_cube(4)
    sample_data.data[0] = np.ma.masked
    sample_data.data[1] = np.nan

    expected = sample_data.copy()
    expected.data[1] = np.ma.masked

    actual = convert_nan_to_masked(sample_data)

    assert actual == expected
