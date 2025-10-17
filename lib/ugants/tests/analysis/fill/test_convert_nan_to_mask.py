# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import numpy as np
from ugants.analysis.fill import convert_nan_to_masked
from ugants.io import load
from ugants.tests import get_data_path


def test_no_nan_no_mask():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")

    actual = convert_nan_to_masked(sample_data)

    assert actual == sample_data


def test_no_nan_one_masked_cell():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")
    sample_data.data[0] = np.ma.masked

    actual = convert_nan_to_masked(sample_data)

    assert actual == sample_data


def test_one_nan_cell_no_mask():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")
    sample_data.data[0] = np.nan

    expected = sample_data.copy()
    expected.data[0] = np.ma.masked

    actual = convert_nan_to_masked(sample_data)

    assert actual == expected


def test_one_nan_one_masked():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")
    sample_data.data[0] = np.ma.masked
    sample_data.data[1] = np.nan

    expected = sample_data.copy()
    expected.data[1] = np.ma.masked

    actual = convert_nan_to_masked(sample_data)

    assert actual == expected
