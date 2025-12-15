# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import re
import tempfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, call

import iris
import numpy as np
import pytest
from iris.cube import CubeList
from iris.exceptions import ConstraintMismatchError

from ugants.io import load
from ugants.regrid.command_line import (
    GridToMeshESMFRegridder,
    Regrid,
    _validate_source_grids,
)
from ugants.tests import get_data_path, stock

OUTPUT_PATH = "/path/to/output.nc"
MESH_NAME = "dynamics"


@pytest.fixture()
def source_path():
    return get_data_path("non_ugrid_data.nc")


@pytest.fixture()
def regular_cubelist():
    """Return a single-element CubeList of regular lat-lon data."""
    return stock.regular_grid_global_cube(144, 192)


@pytest.fixture()
def mesh_path():
    return get_data_path("mesh_C12.nc")


@pytest.fixture()
def mesh_C12_from_file(mesh_path):
    """Load a C12 cubedsphere mesh from disk."""
    return load.mesh(mesh_path, mesh_name=MESH_NAME)


@pytest.fixture()
def input_gridded_to_mesh_weights_path():
    return get_data_path("non_ugrid_data_to_mesh_weights.nc")


class TestCLI:
    @pytest.fixture()
    def default_command(self, source_path, mesh_path):
        """Return default command line arguments for the fill application."""
        return [
            source_path,
            OUTPUT_PATH,
            "--target-mesh",
            mesh_path,
            "--target-mesh-name",
            MESH_NAME,
            "--horizontal-regrid-scheme",
            "conservative",
        ]

    @pytest.fixture()
    def default_app(self, default_command):
        return Regrid.from_command_line(default_command)

    def test_source_loaded(self, default_app, regular_cubelist):
        assert isinstance(default_app.source, CubeList)

    def test_output_path_added(self, default_app):
        assert default_app.output == OUTPUT_PATH

    def test_target_mesh_loaded(self, default_app, mesh_C12_from_file):
        assert default_app.target_mesh == mesh_C12_from_file

    def test_regrid_scheme_added(self, default_app):
        assert default_app.horizontal_regrid_scheme == "conservative"

    def test_default_tolerance_added(self, default_app):
        assert default_app.tolerance == 0.0

    def test_custom_tolerance_added(self, default_command):
        default_command.extend(["--tolerance", "0.5"])
        app = Regrid.from_command_line(default_command)
        assert app.tolerance == 0.5

    def test_invalid_regrid_scheme_fails(self, default_command):
        command = default_command
        command[-1] = "invalid_scheme"
        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            Regrid.from_command_line(command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile(
            "error: argument --horizontal-regrid-scheme: invalid choice: "
            r"'invalid_scheme' \(choose from 'conservative', 'bilinear', 'nearest'\)"
        )
        assert expected_stderr.search(actual_stderr)

    def test_no_mesh_name_given(self, source_path, mesh_path, mesh_C12_from_file):
        """Tests that the mesh will be loaded when not supplied with a mesh name."""
        command = [
            source_path,
            OUTPUT_PATH,
            "--target-mesh",
            mesh_path,
            "--horizontal-regrid-scheme",
            "conservative",
        ]
        app = Regrid.from_command_line(command)
        assert app.target_mesh == mesh_C12_from_file


class TestWeightsCaching:
    def test_check_raises_error_if_input_and_output_weights_provided(
        self,
        regular_cubelist,
        mesh_C12_from_file,
        input_gridded_to_mesh_weights_path,
    ):
        """
        Verifies ValueError raised if both input and output weights used.

        If user provides args for both output_weights and input_weights at the
        same time, an error should be raised.
        """
        with pytest.raises(
            ValueError,
            match="Only one of input_weights and output_weights can be provided",
        ):
            Regrid(
                source=regular_cubelist,
                target_mesh=mesh_C12_from_file,
                horizontal_regrid_scheme="nearest",
                input_weights=input_gridded_to_mesh_weights_path,
                output_weights="synthetic_output_weights_path.nc",
            )

    def test_check_raises_error_if_mesh_file_used_as_input_weights(
        self,
        regular_cubelist,
        mesh_path,
        mesh_C12_from_file,
    ):
        """
        Test attempts to use mesh file as input weights.

        Verifies error raised if mesh file is used for input weights.
        """
        incompatible_input_weights_file = mesh_path

        app = Regrid(
            source=regular_cubelist,
            target_mesh=mesh_C12_from_file,
            horizontal_regrid_scheme="nearest",
            input_weights=incompatible_input_weights_file,
        )
        app.output = "dummy_path"
        with pytest.raises(
            ConstraintMismatchError,
            match=re.escape(
                "Got 0 cubes for constraint Constraint(name='regri"
                "dder_source_field'), expecting 1."
            ),
        ):
            app.run()

    def test_regridder_called_when_output_weights_provided(
        self,
        regular_cubelist,
        mesh_C12_from_file,
    ):
        """Check GridToMeshESMFRegridder used if output_weights provided."""
        app = Regrid(
            source=regular_cubelist,
            target_mesh=mesh_C12_from_file,
            horizontal_regrid_scheme="conservative",
            output_weights="synthetic_output_weights_path.nc",
        )

        regridder_target = "ugants.regrid.command_line.GridToMeshESMFRegridder"
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()
        mock_regridder.assert_called_once()

    def test_regridder_not_called_when_input_weights_provided(
        self,
        regular_cubelist,
        mesh_C12_from_file,
        input_gridded_to_mesh_weights_path,
    ):
        """Check GridToMeshESMFRegridder not used if input_weights provided."""
        app = Regrid(
            source=regular_cubelist,
            target_mesh=mesh_C12_from_file,
            horizontal_regrid_scheme="nearest",
            input_weights=input_gridded_to_mesh_weights_path,
        )

        regridder_target = "ugants.regrid.command_line.GridToMeshESMFRegridder"
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()
        mock_regridder.assert_not_called()

    def test_save_regridder_called(
        self,
        regular_cubelist,
        mesh_C12_from_file,
    ):
        """Check save_regridder functionality.

        Checks save_regridder is called and its args were called in correct order.
        Also verifies an output weights file is generated when output_weights
        path provided.

        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_temporary_data_path = (
                f"{temporary_directory}/temporary_output_data.nc"
            )
            output_temporary_weights_path = (
                f"{temporary_directory}/temporary_output_weights.nc"
            )

            app = Regrid(
                source=regular_cubelist,
                target_mesh=mesh_C12_from_file,
                horizontal_regrid_scheme="conservative",
                output_weights=output_temporary_weights_path,
            )
            app.output = output_temporary_data_path

            mock_saver = "ugants.regrid.command_line.save_regridder"
            # Assert save_regridder being called correctly.
            with mock.patch(mock_saver, autospec=True) as mock_save_regridder:
                app.run()
                app.save()

                mock_save_regridder.assert_called_once_with(
                    app.regridder, app.output_weights
                )
                expected_calls = [call(app.regridder, app.output_weights)]

                mock_save_regridder.assert_has_calls(expected_calls)

            # Assert weights file generated when output_weights path used.
            app.run()
            app.save()
            assert Path(app.output_weights).is_file()

    def test_load_regridder_called(
        self,
        regular_cubelist,
        mesh_C12_from_file,
        input_gridded_to_mesh_weights_path,
    ):
        """Check load_regridder called.

        Check load_regridder is called with the path to synthetic input_weights.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_temporary_data_path = (
                f"{temporary_directory}/temporary_output_data.nc"
            )
            # Mock_regridder to be used as return value for load_regridder,
            # otherwise subsequent validation step will fail in app.run()
            # due to empty mock values.
            mock_regridder = MagicMock(spec=GridToMeshESMFRegridder)
            mock_regridder.mdtol = 0.0
            mock_regridder.method = "nearest"
            mock_regridder.input_weights = input_gridded_to_mesh_weights_path

            app = Regrid(
                source=regular_cubelist,
                target_mesh=mesh_C12_from_file,
                horizontal_regrid_scheme="nearest",
                input_weights=input_gridded_to_mesh_weights_path,
            )
            app.output = output_temporary_data_path

            mock_loader = "ugants.regrid.command_line.load_regridder"
            with mock.patch(
                mock_loader, autospec=True, return_value=mock_regridder
            ) as mock_load_regridder:
                app.run()

                mock_load_regridder.assert_called_once_with(Path(app.input_weights))


class TestWeightsCachingValidation:
    def test_is_netcdf_raises_error_if_nonexistent_input_weights_used(
        self,
        regular_cubelist,
        mesh_C12_from_file,
    ):
        """
        Check is_netcdf function raises error correctly.

        Verifies is_netcdf raises an error when a nonexistent input_weights
        file is used.
        """
        app = Regrid(
            source=regular_cubelist,
            target_mesh=mesh_C12_from_file,
            horizontal_regrid_scheme="nearest",
            input_weights="nonexistent_file_path",
        )

        with pytest.raises(
            ValueError,
            match=("Input file 'nonexistent_file_path'" " must be netCDF."),
        ):
            app.run()

    def test__validate_input_weights_tolerance(
        self,
        regular_cubelist,
        mesh_C12_from_file,
        input_gridded_to_mesh_weights_path,
    ):
        """Test unmatching tolerances raises error.

        Validator should raise error if tolerance in input weights does
        not match the tolerance used on command line.
        """
        app = Regrid(
            source=regular_cubelist,
            target_mesh=mesh_C12_from_file,
            horizontal_regrid_scheme="conservative",
            tolerance=0.6,
            input_weights=input_gridded_to_mesh_weights_path,
        )
        with pytest.raises(
            ValueError,
            match="Tolerance value of input_weights does not match the value"
            " provided on command line.",
        ):
            app.run()

    def test__validate_input_weights_scheme(
        self,
        regular_cubelist,
        mesh_C12_from_file,
        input_gridded_to_mesh_weights_path,
    ):
        """Test unmatching schemes raises error.

        Validator should raise error if scheme used in input weights does
        not match the scheme used on command line.
        """
        app = Regrid(
            source=regular_cubelist,
            target_mesh=mesh_C12_from_file,
            horizontal_regrid_scheme="conservative",
            input_weights=input_gridded_to_mesh_weights_path,
        )
        with pytest.raises(
            ValueError,
            match="Regrid scheme of input_weights does not match the scheme "
            "provided on command line.",
        ):
            app.run()


class TestRegrid:
    @pytest.fixture()
    def default_app(self):
        return Regrid(
            source=stock.regular_grid_global_cube(10, 10),
            target_mesh=stock.regular_lat_lon_mesh(),
            horizontal_regrid_scheme="conservative",
        )

    def test_single_cube_cast_to_cubelist(self):
        """Test that a single cube is converted to a cubelist."""
        source = stock.regular_grid_global_cube(10, 10)
        target_mesh = stock.regular_lat_lon_mesh()
        expected_cubelist = CubeList([source.copy()])
        regrid_app = Regrid(source, target_mesh, "conservative")
        assert regrid_app.source == expected_cubelist

    def test_tolerance_with_nearest_fails(self):
        source = stock.regular_grid_global_cube(10, 10)
        target_mesh = stock.regular_lat_lon_mesh()
        with pytest.raises(
            ValueError,
            match="The 'tolerance' option is not available for regrid scheme 'nearest'",
        ):
            Regrid(
                source=source,
                target_mesh=target_mesh,
                horizontal_regrid_scheme="nearest",
                tolerance=0.5,
            )

    def test_default_regridder_instantiation(self, default_app):
        regridder_target = "ugants.regrid.command_line.GridToMeshESMFRegridder"
        source = stock.regular_grid_global_cube(10, 10)
        target_mesh = stock.regular_lat_lon_mesh()
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            default_app.run()
        mock_regridder.assert_called_once_with(
            src=source,
            tgt=target_mesh,
            method="conservative",
            tgt_location="face",
            mdtol=0,
        )

    def test_regridder_instantiation_with_tolerance(self):
        """Test that the default tolerance can be overridden."""
        source = stock.regular_grid_global_cube(10, 10)
        target_mesh = stock.regular_lat_lon_mesh()

        app = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme="conservative",
            tolerance=0.5,
        )
        regridder_target = "ugants.regrid.command_line.GridToMeshESMFRegridder"
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()
        mock_regridder.assert_called_once_with(
            src=source,
            tgt=target_mesh,
            method="conservative",
            tgt_location="face",
            mdtol=0.5,
        )

    def test_regridder_call(self, default_app):
        """Test that the regridder object is called correctly."""
        source = stock.regular_grid_global_cube(10, 10)
        with mock.patch.object(
            GridToMeshESMFRegridder, "__call__"
        ) as mock_regridder_call:
            default_app.run()
        mock_regridder_call.assert_called_once_with(source)

    def test_multi_source_regrid(self):
        """Test that regridding multiple sources is consistent with regridding single sources."""  # noqa: E501
        source0 = stock.regular_grid_global_cube(10, 10)
        source1 = source0.copy() + 1
        source1.rename("additional_sample_data")
        target_mesh = stock.regular_lat_lon_mesh()

        # regrid each cube individually, creating a regridder for each one
        regridder0 = GridToMeshESMFRegridder(
            source0, target_mesh, method="conservative", tgt_location="face"
        )
        result0 = regridder0(source0)

        regridder1 = GridToMeshESMFRegridder(
            source1, target_mesh, method="conservative", tgt_location="face"
        )
        result1 = regridder1(source1)

        expected_results = CubeList([result0, result1])

        # regrid both cubes using the regrid app
        sources = CubeList([source0, source1])
        app = Regrid(
            source=sources,
            target_mesh=target_mesh,
            horizontal_regrid_scheme="conservative",
        )
        app.run()
        assert app.results == expected_results

    def test_multi_source_regrid_different_vertical(self):
        """Test that regridding multiple sources is consistent with regridding single sources.

        In this case, the sources have the same horizontal grid, but only one of them
        has a vertical coordinate.
        """  # noqa: E501
        target_mesh = stock.regular_lat_lon_mesh()
        # source cube with horizontal dimensions only
        source0 = stock.regular_grid_global_cube(10, 10)

        # create a second source cube with a vertical dimension,
        # and identical horizontal grid
        source1a = source0.copy() + 1
        source1b = source0.copy() + 2
        vertical_coord = iris.coords.DimCoord([0, 1], "model_level_number")
        source1a.add_aux_coord(vertical_coord[0])
        source1b.add_aux_coord(vertical_coord[1])
        source1 = iris.cube.CubeList([source1a, source1b]).merge_cube()

        sources = CubeList([source0, source1])
        assert _validate_source_grids(sources) is None

        # regrid each cube individually, creating a regridder for each one
        regridder0 = GridToMeshESMFRegridder(
            source0, target_mesh, method="conservative", tgt_location="face"
        )
        result0 = regridder0(source0)

        regridder1 = GridToMeshESMFRegridder(
            source1, target_mesh, method="conservative", tgt_location="face"
        )
        result1 = regridder1(source1)

        expected_results = CubeList([result0, result1])

        # regrid both cubes using the regrid app
        app = Regrid(
            source=sources,
            target_mesh=target_mesh,
            horizontal_regrid_scheme="conservative",
        )
        app.run()
        assert app.results == expected_results

    def test_different_source_grids_raises_error(self):
        """Test that an error is raised when cubes have different horizontal grids."""
        source0 = stock.regular_grid_global_cube(10, 10)
        source1 = stock.regular_grid_global_cube(11, 11)
        source_cubes = CubeList([source0, source1])
        target_mesh = stock.regular_lat_lon_mesh()
        with pytest.raises(
            ValueError, match="Not all source cubes have the same horizontal grid."
        ):
            Regrid(source_cubes, target_mesh, "conservative")


@pytest.mark.parametrize("scheme", ["conservative", "bilinear"])
class TestConsistencyInResults:
    """Unmasked cells in the output should have the same data, regardless of tolerance."""  # noqa: E501

    def test_source_fully_unmasked(self, scheme):
        """Results should be identical.

        Since all cells in the source are unmasked, the tolerance option
        should have no effect on the results.
        """
        source = stock.regular_grid_global_cube(10, 10)
        target_mesh = stock.regular_lat_lon_mesh()
        # Run regrid without tolerance
        app_without_tolerance = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme=scheme,
        )
        app_without_tolerance.run()
        regridded_without_tolerance = app_without_tolerance.results

        # Run regrid with tolerance
        app_with_tolerance = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme=scheme,
            tolerance=0.5,
        )
        app_with_tolerance.run()
        regridded_with_tolerance = app_with_tolerance.results

        assert regridded_with_tolerance == regridded_without_tolerance

    def test_source_partially_masked(self, scheme):
        """Results in the overlapping unmasked regions should be equivalent.

        Cells that are unmasked in the regrid results both
        with and without tolerance, should be the same.
        """
        target_mesh = stock.regular_lat_lon_mesh()

        # Mask some cells in the source data.
        source = stock.regular_grid_global_cube(10, 10)
        source.data[(1, 3, 5), (2, 4, 6)] = np.ma.masked

        # Run regrid without tolerance
        app_without_tolerance = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme=scheme,
        )
        app_without_tolerance.run()
        assert len(app_without_tolerance.results) == 1
        regridded_without_tolerance = app_without_tolerance.results[0]

        # Run regrid with tolerance
        app_with_tolerance = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme=scheme,
            tolerance=0.5,
        )
        app_with_tolerance.run()
        assert len(app_with_tolerance.results) == 1
        regridded_with_tolerance = app_with_tolerance.results[0]

        unmasked_in_both = np.logical_and(
            regridded_without_tolerance.data.mask, regridded_with_tolerance.data.mask
        )
        np.testing.assert_array_equal(
            regridded_without_tolerance.data[unmasked_in_both],
            regridded_with_tolerance.data[unmasked_in_both],
        )

    def test_increased_tolerance_decreases_masked_cells(self, scheme):
        """The number of masked cells in the output should be fewer with tolerance than without.

        Consider the example below. The diagram represents a single target cell.
        A small region of the target cell overlaps with
        masked cells in the source (represented by X).
        +-------+
        |       |  If tolerance = 0, then the target cell is masked.
        |     XX|  If tolerance = 0.5, then the target cell is unmasked.
        |     XX|
        +-------+
        """  # noqa: E501
        target_mesh = stock.regular_lat_lon_mesh()

        # Mask some cells in the source data.
        source = stock.regular_grid_global_cube(10, 10)
        source.data[(1, 3, 5), (2, 4, 6)] = np.ma.masked

        # Run regrid without tolerance
        app_without_tolerance = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme=scheme,
        )
        app_without_tolerance.run()
        assert len(app_without_tolerance.results) == 1
        regridded_without_tolerance = app_without_tolerance.results[0]
        mask_without_tolerance = regridded_without_tolerance.data.mask

        # Run regrid with tolerance
        app_with_tolerance = Regrid(
            source=source,
            target_mesh=target_mesh,
            horizontal_regrid_scheme=scheme,
            tolerance=0.5,
        )
        app_with_tolerance.run()
        assert len(app_with_tolerance.results) == 1
        regridded_with_tolerance = app_with_tolerance.results[0]
        mask_with_tolerance = regridded_with_tolerance.data.mask

        assert mask_with_tolerance.sum() < mask_without_tolerance.sum()
