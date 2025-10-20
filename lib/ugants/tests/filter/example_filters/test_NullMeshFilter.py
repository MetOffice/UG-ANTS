# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np

from ugants.filter.example_filters import NullMeshFilter
from ugants.tests.stock import sample_mesh_cube


def test():
    """Check that you can create and use the filter, and that result == input."""
    test_cube = sample_mesh_cube()
    # Patch in some dummy data, just to exercise the computation.
    test_data = np.random.uniform(0, 1, test_cube.core_data().shape)
    test_cube.data = test_data
    filter_instance = NullMeshFilter(
        test_cube,
    )

    result = filter_instance(test_cube)

    assert result == test_cube
