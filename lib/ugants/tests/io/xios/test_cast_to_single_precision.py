# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import dask.array as da
import iris.cube
import numpy as np
import pytest
from ugants.io.xios_command_line import cast_to_single_precision


class TestRealData:
    @pytest.fixture()
    def real_cubes(self):
        """Create a cubelist containing three cubes with real data.

        The cubes have data types float64, float32, and int64.
        """
        float64_cube = iris.cube.Cube(
            da.ma.masked_array([0], dtype="float64"), long_name="double_data"
        )
        float32_cube = iris.cube.Cube(
            da.ma.masked_array([0], dtype="float32"), long_name="single_data"
        )
        int64_cube = iris.cube.Cube(
            da.ma.masked_array([0], dtype="int64"), long_name="integer_data"
        )
        cubes = iris.cube.CubeList([float64_cube, float32_cube, int64_cube])
        return cubes

    @pytest.mark.filterwarnings(
        "ignore:The following variables are not double precision"
    )
    def test_data_type_cast(self, real_cubes):
        """Test that float64 data has been cast to float32.

        The float64 cube should be cast to float32, the others should not be cast.
        """
        expected_dtypes = [np.dtype("float32"), np.dtype("float32"), np.dtype("int64")]
        cast_cubes = cast_to_single_precision(real_cubes)
        actual_dtypes = [cube.dtype for cube in cast_cubes]

        assert expected_dtypes == actual_dtypes

    def test_warning_raised(self, real_cubes):
        """Test that a warning is raised when non-double data is passed."""
        expected_msg = (
            "The following variables are not double precision, and will not be cast to "
            "single precision: single_data has dtype float32, integer_data has dtype "
            "int64."
        )
        with pytest.warns(UserWarning, match=expected_msg):
            cast_to_single_precision(real_cubes)


class TestLazyData:
    @pytest.fixture()
    def lazy_cubes(self):
        """Create a cubelist containing three cubes with lazy data.

        The cubes have data types float64, float32, and int64.
        """
        float64_cube = iris.cube.Cube(
            da.ma.masked_array([0], dtype="float64"), long_name="double_data"
        )
        float32_cube = iris.cube.Cube(
            da.ma.masked_array([0], dtype="float32"), long_name="single_data"
        )
        int64_cube = iris.cube.Cube(
            da.ma.masked_array([0], dtype="int64"), long_name="integer_data"
        )
        cubes = iris.cube.CubeList([float64_cube, float32_cube, int64_cube])
        return cubes

    @pytest.mark.filterwarnings(
        "ignore:The following variables are not double precision"
    )
    def test_data_type_cast(self, lazy_cubes):
        """Test that the data type of the float64 cube has been cast to float32."""
        expected_dtypes = [np.dtype("float32"), np.dtype("float32"), np.dtype("int64")]
        cast_cubes = cast_to_single_precision(lazy_cubes)
        actual_dtypes = [cube.dtype for cube in cast_cubes]

        assert expected_dtypes == actual_dtypes

    @pytest.mark.filterwarnings(
        "ignore:The following variables are not double precision"
    )
    def test_laziness_maintained(self, lazy_cubes):
        """Ensure that laziness is maintained when casting dtype."""
        assert all(cube.has_lazy_data() for cube in lazy_cubes)
        cast_cubes = cast_to_single_precision(lazy_cubes)
        assert all(cube.has_lazy_data() for cube in cast_cubes)
        assert all(cube.has_lazy_data() for cube in lazy_cubes)

    def test_warning_raised(self, lazy_cubes):
        """Test that a warning is raised when non-double data is passed."""
        expected_msg = (
            "The following variables are not double precision, and will not be cast to "
            "single precision: single_data has dtype float32, integer_data has dtype "
            "int64."
        )
        with pytest.warns(UserWarning, match=expected_msg):
            cast_to_single_precision(lazy_cubes)
