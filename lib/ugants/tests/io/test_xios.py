# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import unittest.mock

import iris.cube
import netCDF4
import numpy as np
import pytest

import ugants
import ugants.tests
import ugants.tests.stock
from ugants.io.xios_command_line import ConvertToXIOS


@pytest.mark.parametrize("cast_to_single", (True, False))
class TestConvertToXIOS:
    def test_long_name_workaround_on_disk(
        self, cast_to_single, temporary_filepaths_function
    ):
        """Assert that long name is set on disk even if there is a standard name."""
        expected = "surface_altitude"

        source_cubes = ugants.io.load.ugrid(
            ugants.tests.get_data_path("data_C4.nc"), "sample_data"
        )
        assert len(source_cubes) == 1
        source_cube = source_cubes[0]
        source_cube.rename("surface_altitude")
        assert source_cube.long_name is None
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(source_cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        actual = dataset["surface_altitude"].long_name

        assert actual == expected

    def test_long_name_workaround_on_cube(
        self, cast_to_single, temporary_filepaths_function
    ):
        """Assert that long name is not set on cube even when it is set on disk."""
        expected = None

        source_cubes = ugants.io.load.ugrid(
            ugants.tests.get_data_path("data_C4.nc"), "sample_data"
        )
        assert len(source_cubes) == 1
        source_cube = source_cubes[0]
        source_cube.rename("surface_altitude")
        assert source_cube.long_name is None
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(source_cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        actual = source_cube.long_name

        assert actual == expected

    def test_default_online_operation_workaround_on_disk(
        self, cast_to_single, temporary_filepaths_function
    ):
        expected = "once"

        source_cubes = ugants.io.load.ugrid(
            ugants.tests.get_data_path("data_C4.nc"), "sample_data"
        )
        assert len(source_cubes) == 1
        source_cube = source_cubes[0]
        source_cube.attributes.pop("online_operation")
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(source_cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        actual = dataset["sample_data"].online_operation

        assert actual == expected

    def test_user_specified_online_operation_on_disk(
        self, cast_to_single, temporary_filepaths_function
    ):
        expected = "foo"

        source_cubes = ugants.io.load.ugrid(
            ugants.tests.get_data_path("data_C4.nc"), "sample_data"
        )
        assert len(source_cubes) == 1
        source_cube = source_cubes[0]
        source_cube.attributes["online_operation"] = "foo"
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(source_cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        actual = dataset["sample_data"].online_operation

        assert actual == expected

    def test_default_online_operation_not_set_on_cube(
        self, cast_to_single, temporary_filepaths_function
    ):
        source_cubes = ugants.io.load.ugrid(
            ugants.tests.get_data_path("data_C4.nc"), "sample_data"
        )
        assert len(source_cubes) == 1
        source_cube = source_cubes[0]
        source_cube.attributes.pop("online_operation")
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(source_cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()

        assert "online_operation" not in source_cube.attributes

    def test_user_specified_online_operation_not_changed_on_cube(
        self, cast_to_single, temporary_filepaths_function
    ):
        expected = "foo"

        source_cubes = ugants.io.load.ugrid(
            ugants.tests.get_data_path("data_C4.nc"), "sample_data"
        )
        assert len(source_cubes) == 1
        source_cube = source_cubes[0]
        source_cube.attributes["online_operation"] = "foo"
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(source_cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        actual = source_cube.attributes["online_operation"]

        assert actual == expected

    def test_run_calls_long_name_workaround(self, cast_to_single):
        cube = ugants.tests.stock.mesh_cube()
        converter = ConvertToXIOS(cube, cast_to_single)
        with unittest.mock.patch(
            "ugants.io.xios_command_line.ConvertToXIOS.apply_long_name_workaround"
        ) as mock_long_name_workaround:
            converter.run()
        assert mock_long_name_workaround.called_once()

    def test_run_calls_online_operation_workaround(self, cast_to_single):
        cube = ugants.tests.stock.mesh_cube()
        converter = ConvertToXIOS(cube, cast_to_single)
        with unittest.mock.patch(
            "ugants.io.xios_command_line.ConvertToXIOS.apply_online_operation_workaround"
        ) as mock_online_operation_workaround:
            converter.run()
        assert mock_online_operation_workaround.called_once()

    def test_run_calls_apply_dimension_order_workaround(self, cast_to_single):
        cube = ugants.tests.stock.mesh_cube()
        converter = ConvertToXIOS(cube, cast_to_single)
        with unittest.mock.patch(
            "ugants.io.xios_command_line.ConvertToXIOS.apply_dimension_order_workaround"
        ) as mock_online_operation_workaround:
            converter.run()
        assert mock_online_operation_workaround.called_once()

    def test_dimension_order_workaround_on_disk(
        self, cast_to_single, temporary_filepaths_function
    ):
        cube = ugants.tests.stock.mesh_cube()
        cube = ugants.tests.add_axis(cube, "pseudo_level")
        cube = ugants.tests.add_axis(cube, "model_level", axis="z")
        cube = ugants.tests.add_axis(cube, "time", axis="t")
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        actual = [
            dataset.dimensions[dimension].name for dimension in dataset.dimensions
        ]
        # There's some additional dimensions to do with the mesh that we don't
        # care about - we need to ensure that the "other, time, vertical
        # levels" part is in that order (as per CF and XIOS), but ignore the
        # rest.
        pseudo_level_index = actual.index("pseudo_level")
        time_index = actual.index("time")
        model_level_index = actual.index("model_level")
        assert pseudo_level_index < time_index < model_level_index

    def test_dimension_order_workaround_preserves_other_dimensions_on_disk(
        self, cast_to_single, temporary_filepaths_function
    ):
        cube = ugants.tests.stock.mesh_cube()
        cube = ugants.tests.add_axis(cube, "bar")
        cube = ugants.tests.add_axis(cube, "foo")
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(cube, cast_to_single)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        actual = [
            dataset.dimensions[dimension].name for dimension in dataset.dimensions
        ]
        # There's some additional dimensions to do with the mesh that we don't
        # care about - we need to ensure that "foo" and "bar" are in that
        # order, but ignore the rest.
        foo_index = actual.index("foo")
        bar_index = actual.index("bar")
        assert foo_index < bar_index


@pytest.mark.filterwarnings("ignore:Fill value not known for the following variables")
class TestPrecision:
    """Tests for behaviour dependent on the precision of the data being saved."""

    @pytest.fixture()
    def mixed_precision_cubelist(self):
        """Create a cubelist with three mesh cubes with different data types.

        * double_data: float64
        * single_data: float32
        * integer_data: int64
        """
        float64_cube = ugants.tests.stock.mesh_cube()
        float64_cube.rename("double_data")
        float64_cube = float64_cube.copy(float64_cube.core_data().astype("float64"))

        float32_cube = ugants.tests.stock.mesh_cube()
        float32_cube.rename("single_data")
        float32_cube = float32_cube.copy(float32_cube.core_data().astype("float32"))

        int64_cube = ugants.tests.stock.mesh_cube()
        int64_cube.rename("integer_data")
        int64_cube = int64_cube.copy(int64_cube.core_data().astype("int64"))

        cubes = iris.cube.CubeList([float64_cube, float32_cube, int64_cube])
        return cubes

    def test_fill_value_workaround_on_disk_nocast(
        self, temporary_filepaths_function, mixed_precision_cubelist
    ):
        """Test that the _FillValue attribute has been set correctly when data is not cast."""  # noqa: E501
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(mixed_precision_cubelist, cast_to_single=False)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        double_data_fill_value = dataset["double_data"]._FillValue
        single_data_fill_value = dataset["single_data"]._FillValue

        assert double_data_fill_value == np.nan_to_num(np.NINF)
        assert single_data_fill_value == -np.float32(32768 * 32768)
        assert not hasattr(dataset["integer_data"], "_FillValue")

    @pytest.mark.filterwarnings(
        "ignore:The following variables are not double precision, and will not be cast to single precision"  # noqa: E501
    )
    def test_fill_value_workaround_on_disk_cast(
        self, temporary_filepaths_function, mixed_precision_cubelist
    ):
        """Test that the _FillValue attribute has been set correctly when data is cast."""  # noqa: E501
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(mixed_precision_cubelist, cast_to_single=True)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        double_data_fill_value = dataset["double_data"]._FillValue
        single_data_fill_value = dataset["single_data"]._FillValue

        assert double_data_fill_value == -np.float32(32768 * 32768)
        assert single_data_fill_value == -np.float32(32768 * 32768)
        assert not hasattr(dataset["integer_data"], "_FillValue")

    def test_dtypes_on_disk_nocast(
        self, temporary_filepaths_function, mixed_precision_cubelist
    ):
        """Test that data has not been cast to single precision on save."""
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(mixed_precision_cubelist, cast_to_single=False)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        assert dataset["double_data"].dtype == np.dtype("float64")
        assert dataset["single_data"].dtype == np.dtype("float32")
        assert dataset["integer_data"].dtype == np.dtype("int64")

    @pytest.mark.filterwarnings(
        "ignore:The following variables are not double precision, and will not be cast to single precision"  # noqa: E501
    )
    def test_dtypes_on_disk_cast(
        self, temporary_filepaths_function, mixed_precision_cubelist
    ):
        """Test that data has been cast to single precision on save."""
        output_path = temporary_filepaths_function(suffix=".nc")
        converter = ConvertToXIOS(mixed_precision_cubelist, cast_to_single=True)
        converter.output = output_path
        converter.run()
        converter.save()
        dataset = netCDF4.Dataset(output_path)
        assert dataset["double_data"].dtype == np.dtype("float32")
        assert dataset["single_data"].dtype == np.dtype("float32")
        assert dataset["integer_data"].dtype == np.dtype("int64")


def test_get_fill_values():
    """Test that appropriate fill values are identified according to data type."""
    float64_cube = iris.cube.Cube(np.ma.array([0], dtype="float64"))
    float32_cube = iris.cube.Cube(np.ma.array([0], dtype="float32"))
    int64_cube = iris.cube.Cube(
        np.ma.array([0], dtype="int64"), long_name="integer_64_data"
    )
    int32_cube = iris.cube.Cube(
        np.ma.array([0], dtype="int32"), long_name="integer_32_data"
    )
    cubes = iris.cube.CubeList([float64_cube, float32_cube, int64_cube, int32_cube])
    expected_fill_values = [
        np.nan_to_num(np.NINF),
        -np.float32(32768 * 32768),
        None,
        None,
    ]

    expected_msg = (
        "Fill value not known for the following variables: "
        "integer_64_data has dtype int64, integer_32_data has dtype int32. "
        "Using default fill values for these dtypes."
    )
    with pytest.warns(match=expected_msg):
        actual_fill_values = ConvertToXIOS.get_fill_values(cubes)

    assert expected_fill_values == actual_fill_values
