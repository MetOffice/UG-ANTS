# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.


import numpy as np
import pytest
from numpy.testing import assert_allclose
from ugants.analysis.coord_transforms import convert_to_cartesian
from ugants.tests.stock import cubedsphere_cube

pytestmark = pytest.mark.filterwarnings(
    "ignore:Assuming a spherical geocentric coordinate system for conversion to "
    "3D cartesian coordinates. If the provided cube is not defined on this "
    "coordinate system then unexpected results may occur.:UserWarning",
)


@pytest.fixture()
def sample_data():
    return cubedsphere_cube(4)


def test_radii(sample_data):
    point_cloud = convert_to_cartesian(sample_data)
    # calculate radius of each point
    radii = (point_cloud**2).sum(axis=1) ** 0.5
    assert_allclose(actual=radii, desired=1.0)


def test_individual_points(sample_data):
    """Check a handful of points in the sample data.

    >>> sample_data.coord("longitude").points[[1, 30, 42, 50, 87]]
    array([-143.13010235,   36.86989765,  -75.96375653,   75.96375653,
           -18.43494882])
    >>> sample_data.coord("latitude").points[[1, 30, 42, 50, 87]]
    array([-11.30993247, -11.30993247,  13.63302223,  36.03989343,
    """
    point_cloud = convert_to_cartesian(sample_data)
    actual_subset = point_cloud[[1, 30, 42, 50, 87]]
    expected_subset = np.array(
        [
            [-0.78446454, -0.58834841, -0.19611614],
            [0.78446454, 0.58834841, -0.19611614],
            [0.23570226, -0.94280904, 0.23570226],
            [0.19611614, 0.78446454, 0.58834841],
            [0.58834841, -0.19611614, 0.78446454],
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
