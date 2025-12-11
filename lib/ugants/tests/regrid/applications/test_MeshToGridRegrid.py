# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

import iris.cube
import pytest
from esmf_regrid.experimental.unstructured_scheme import (
    MeshToGridESMFRegridder,
)
from iris.cube import CubeList

from ugants.exceptions import ProvisionalWarning
from ugants.io.load import ugrid
from ugants.io.save import PROVISIONAL_WARNING_MESSAGE
from ugants.regrid.applications import MeshToGridRegrid
from ugants.tests import get_data_path
from ugants.tests.io.save.test_ugrid import _get_netcdf_global_attribute, _MockDateTime
from ugants.tests.stock import regular_grid_global_cube

OUTPUT_PATH = "/path/to/output.nc"


@pytest.fixture()
def source_path():
    """Return path to mesh file of C4 resolution."""
    return get_data_path("data_C4.nc")


@pytest.fixture()
def ugrid_cubelist(source_path):
    """Return cubelist of two elements."""
    return ugrid(source_path)


@pytest.fixture()
def single_element_cubelist(ugrid_cubelist):
    """Return single element "sample_data" from the source CubeList."""
    return iris.cube.CubeList.extract(ugrid_cubelist, "sample_data")


@pytest.fixture()
def target_path():
    """Return path to lat-lon file of N96 resolution."""
    return get_data_path("non_ugrid_data.nc")


@pytest.fixture()
def regular_grid_global_cube_cubelist():
    """Return a cubelist containing a single lat-lon cube."""
    return CubeList([regular_grid_global_cube(144, 192)])


@pytest.fixture()
def input_weights():
    """
    Return path to output path of weights.

    Weights file generated from a run of mesh to grid using data_C4.nc as source
    and non_ugrid_data.nc as target.
    """
    return get_data_path("mesh_to_grid_output_weights_C4_to_n96.nc")


class TestCLI:
    @pytest.fixture()
    def default_command(self, source_path, target_path):
        """Return default command line arguments for mesh to grid regrid application."""
        return [
            source_path,
            target_path,
            OUTPUT_PATH,
            "--horizontal-regrid-scheme",
            "conservative",
        ]

    @pytest.fixture()
    def default_app(self, default_command):
        return MeshToGridRegrid.from_command_line(default_command)

    def test_source_loaded(self, default_app, ugrid_cubelist):
        assert default_app.source == ugrid_cubelist

    def test_output_path_added(self, default_app):
        assert default_app.output == OUTPUT_PATH

    def test_target_cube(self, default_app):
        assert isinstance(default_app.target_grid, CubeList)

    def test_regrid_scheme_added(self, default_app):
        assert default_app.horizontal_regrid_scheme == "conservative"

    def test_invalid_regrid_scheme_fails(self, default_command):
        command = default_command
        command[-1] = "invalid_scheme"

        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            MeshToGridRegrid.from_command_line(command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile(
            "error: argument --horizontal-regrid-scheme: invalid choice: "
            r"'invalid_scheme' \(choose from 'conservative', 'bilinear', 'nearest'\)"
        )
        assert expected_stderr.search(actual_stderr)


class TestWeightsCaching:
    def test_check_raises_error_if_input_and_output_weights_provided(
        self,
        single_element_cubelist,
        regular_grid_global_cube_cubelist,
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
            MeshToGridRegrid(
                source=single_element_cubelist,
                target_grid=regular_grid_global_cube_cubelist,
                horizontal_regrid_scheme="conservative",
                input_weights="synthetic_input_weights_path",
                output_weights="synthetic_output_weights_path.nc",
            )

    def test_regridder_called_when_output_weights_provided(
        self,
        single_element_cubelist,
        regular_grid_global_cube_cubelist,
    ):
        """Check MeshToGridESMFRegridder used if output_weights provided."""
        app = MeshToGridRegrid(
            source=single_element_cubelist,
            target_grid=regular_grid_global_cube_cubelist,
            horizontal_regrid_scheme="conservative",
            output_weights="synthetic_output_weights_path.nc",
        )

        regridder_target = "ugants.regrid.applications.MeshToGridESMFRegridder"
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()
        mock_regridder.assert_called_once()

    def test_regridder_not_called_when_input_weights_provided(
        self,
        single_element_cubelist,
        regular_grid_global_cube_cubelist,
        input_weights,
    ):
        """Check MeshToGridESMFRegridder not used if input_weights provided."""
        app = MeshToGridRegrid(
            source=single_element_cubelist,
            target_grid=regular_grid_global_cube_cubelist,
            horizontal_regrid_scheme="conservative",
            input_weights=input_weights,
        )

        regridder_target = "ugants.regrid.applications.MeshToGridESMFRegridder"
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()
        mock_regridder.assert_not_called()

    def test_save_regridder_called(
        self,
        single_element_cubelist,
        regular_grid_global_cube_cubelist,
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

            app = MeshToGridRegrid(
                source=single_element_cubelist,
                target_grid=regular_grid_global_cube_cubelist,
                horizontal_regrid_scheme="conservative",
                output_weights=output_temporary_weights_path,
            )
            app.output = output_temporary_data_path

            mock_saver = "ugants.regrid.applications.save_regridder"
            # Assert save_regridder being called correctly.
            with mock.patch(mock_saver, autospec=True) as mock_save_regridder:
                app.run()
                app.save()

                mock_save_regridder.assert_called_once_with(
                    app.regridder, app.output_weights
                )
                expected_calls = [mock.call(app.regridder, app.output_weights)]

                mock_save_regridder.assert_has_calls(expected_calls)

            # Assert weights file generated when output_weights path used.
            app.run()
            app.save()
            assert Path(app.output_weights).is_file()

    def test_load_regridder_called(
        self,
        single_element_cubelist,
        regular_grid_global_cube_cubelist,
        input_weights,
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
            mock_regridder = mock.MagicMock(spec=MeshToGridESMFRegridder)
            mock_regridder.mdtol = 0.0
            mock_regridder.method = "conservative"
            mock_regridder.input_weights = input_weights

            app = MeshToGridRegrid(
                source=single_element_cubelist,
                target_grid=regular_grid_global_cube_cubelist,
                horizontal_regrid_scheme="conservative",
                input_weights=input_weights,
            )
            app.output = output_temporary_data_path

            mock_loader = "ugants.regrid.applications.load_regridder"
            with mock.patch(
                mock_loader, autospec=True, return_value=mock_regridder
            ) as mock_load_regridder:
                app.run()

                mock_load_regridder.assert_called_once_with(Path(app.input_weights))


class TestWeightsCachingValidation:
    def test_is_netcdf_raises_error_if_nonexistent_input_weights_used(
        self,
        single_element_cubelist,
        regular_grid_global_cube_cubelist,
    ):
        """
        Check is_netcdf function raises error correctly.

        Verifies is_netcdf raises an error when a nonexistent input_weights
        file is used.
        """
        app = MeshToGridRegrid(
            source=single_element_cubelist,
            target_grid=regular_grid_global_cube_cubelist,
            horizontal_regrid_scheme="conservative",
            input_weights="nonexistent_file_path",
        )

        with pytest.raises(
            OSError,
            match=("One or more of the files specified did not exist:"),
        ):
            app.run()

    def test__validate_input_weights_scheme(
        self, single_element_cubelist, regular_grid_global_cube_cubelist, input_weights
    ):
        """Test unmatching schemes raises error.

        Validator should raise error if scheme used in input weights does
        not match the scheme used on command line.
        """
        app = MeshToGridRegrid(
            source=single_element_cubelist,
            target_grid=regular_grid_global_cube_cubelist,
            horizontal_regrid_scheme="bilinear",
            input_weights=input_weights,
        )
        with pytest.raises(
            ValueError,
            match="Regrid scheme of input_weights does not match the scheme "
            "provided on command line.",
        ):
            app.run()


class TestRegrid:
    @pytest.fixture()
    def default_app(self, single_element_cubelist, regular_grid_global_cube_cubelist):
        return MeshToGridRegrid(
            source=single_element_cubelist,
            target_grid=regular_grid_global_cube_cubelist,
            horizontal_regrid_scheme="conservative",
        )

    def test_regridder_call(self, default_app, single_element_cubelist):
        with mock.patch.object(
            MeshToGridESMFRegridder, "__call__"
        ) as mock_regridder_call:
            default_app.run()
        mock_regridder_call.assert_called_once_with(single_element_cubelist[0])

    def test_multi_source_fail(self, ugrid_cubelist, regular_grid_global_cube_cubelist):
        sources = CubeList([ugrid_cubelist[0], ugrid_cubelist[0] + 1])
        app = MeshToGridRegrid(
            source=sources,
            target_grid=regular_grid_global_cube_cubelist,
            horizontal_regrid_scheme="conservative",
        )
        with pytest.raises(ValueError, match="Source contained 2 cubes, expected 1."):
            app.run()

    def test_multiple_target_fail(
        self, ugrid_cubelist, regular_grid_global_cube_cubelist
    ):
        targets = CubeList(
            [
                regular_grid_global_cube_cubelist[0],
                regular_grid_global_cube_cubelist[0] + 1,
            ]
        )
        app = MeshToGridRegrid(
            source=iris.cube.CubeList(
                [
                    ugrid_cubelist[0],
                ]
            ),
            target_grid=targets,
            horizontal_regrid_scheme="conservative",
        )
        with pytest.raises(
            ValueError, match="Target Grid contained 2 cubes, expected 1."
        ):
            app.run()


class TestSave:
    @pytest.fixture()
    def temp_output_file(self, temporary_filepaths_function):
        path = temporary_filepaths_function(suffix=".nc")
        return path

    @pytest.fixture()
    def app(self, ugrid_cubelist, regular_grid_global_cube_cubelist, temp_output_file):
        app = MeshToGridRegrid(
            ugrid_cubelist, regular_grid_global_cube_cubelist, "conservative"
        )
        app.output = temp_output_file
        app.results = regular_grid_global_cube_cubelist[0]
        return app

    def test_written_to_disk(self, app, temp_output_file):
        assert not temp_output_file.exists()
        app.save()
        assert temp_output_file.exists()

    def test_provisional_warning(self, app):
        with pytest.warns(ProvisionalWarning) as warnings:
            app.save()

        found_warning = False
        # There's one warning from ug-ants, and at least one warning from
        # iris.  We only care that our warning is included in the list of
        # warnings - we don't care about other warnings, or whether our
        # warning is first, last or somewhere in the middle:
        for warning in warnings:
            if str(warning.message) == PROVISIONAL_WARNING_MESSAGE:
                found_warning = True
        assert found_warning

    def test_provisional_attribute_on_disk(self, app, temp_output_file):
        assert "ugants_status" not in app.results.attributes
        app.save()
        actual = _get_netcdf_global_attribute(temp_output_file, "ugants_status")
        assert actual == PROVISIONAL_WARNING_MESSAGE

    def test_suite_provenance_attribute_on_disk(
        self, app, temp_output_file, monkeypatch
    ):
        expected = "r1234_branch@5678"

        assert "suite_provenance" not in app.results.attributes
        monkeypatch.setenv("SUITE_PROVENANCE", expected)
        app.save()
        actual = _get_netcdf_global_attribute(temp_output_file, "suite_provenance")

        assert actual == expected

    def test_none_suite_provenance_on_disk(self, app, temp_output_file, monkeypatch):
        expected = "None"

        assert "suite_provenance" not in app.results.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        app.save()
        actual = _get_netcdf_global_attribute(temp_output_file, "suite_provenance")

        assert actual == expected

    def test_create_history_attribute_on_disk(self, app, temp_output_file):
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar"

        del app.results.attributes["history"]
        assert "history" not in app.results.attributes

        with (
            unittest.mock.patch(
                "ugants.utils.cube.datetime", _MockDateTime(reference_date)
            ),
            unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]),
        ):
            app.save()
            actual = _get_netcdf_global_attribute(temp_output_file, "history")

        assert actual == expected

    def test_append_history_attribute_on_disk(self, app, temp_output_file):
        """Test that history is prepended to a pre-existing history attribute."""
        app.results.attributes["history"] = "old_history"
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar\nold_history"

        with (
            unittest.mock.patch(
                "ugants.utils.cube.datetime", _MockDateTime(reference_date)
            ),
            unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]),
        ):
            app.save()
            actual = _get_netcdf_global_attribute(temp_output_file, "history")

        assert actual == expected
