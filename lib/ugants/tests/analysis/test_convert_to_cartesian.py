# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.


import numpy as np
import pytest
from numpy.testing import assert_allclose
from ugants.analysis.coord_transforms import convert_to_cartesian
from ugants.io import load
from ugants.tests import get_data_path

pytestmark = pytest.mark.filterwarnings(
    "ignore:Assuming a spherical geocentric coordinate system for conversion to "
    "3D cartesian coordinates. If the provided cube is not defined on this "
    "coordinate system then unexpected results may occur.:UserWarning",
)


@pytest.fixture()
def sample_data():
    data_C4 = load.ugrid(get_data_path("data_C4.nc"))
    sample_data = data_C4.extract_cube("sample_data")
    return sample_data


def test_radii(sample_data):
    point_cloud = convert_to_cartesian(sample_data)
    # calculate radius of each point
    radii = (point_cloud**2).sum(axis=1) ** 0.5
    assert_allclose(actual=radii, desired=1.0, atol=0.01)


def test_individual_points(sample_data):
    """Check a handful of points in the sample data.

    >>> sample_data.coord("longitude").points[[1, 30, 42, 50, 87]]
    masked_array(data=[348.62056535, 101.37943465, 191.27974683, 281.37943465,
                       163.28214317],
                 mask=False,
                 fill_value=1e+20)
    >>> sample_data.coord("latitude").points[[1, 30, 42, 50, 87]]
    masked_array(data=[ 33.30129185, -33.30129185, -11.06729582,  33.30129185,
                       -55.02127549],
                 mask=False,
                 fill_value=1e+20)
    """
    point_cloud = convert_to_cartesian(sample_data)
    actual_subset = point_cloud[[1, 30, 42, 50, 87]]
    expected_subset = np.array(
        [
            [0.81936497, -0.16490693, 0.54904166],
            [-0.16490693, 0.81936497, -0.54904166],
            [-0.96244549, -0.19196182, -0.19196182],
            [0.16490693, -0.81936497, 0.54904166],
            [-0.54904166, 0.16490693, -0.81936497],
        ]
    )
    np.testing.assert_array_almost_equal(actual_subset, expected_subset)


def test_geocentric_warning(sample_data):
    """Test that a warning is raised for assuming a geocentric a coordinate system."""
    with pytest.warns(
        UserWarning,
        match="Assuming a spherical geocentric coordinate system for conversion to "
        "3D cartesian coordinates. If the provided cube is not defined on this "
        "coordinate system then unexpected results may occur.",
    ):
        convert_to_cartesian(sample_data)
