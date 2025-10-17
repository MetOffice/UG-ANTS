# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import re
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

import iris.cube
import pytest
from esmf_regrid.experimental.unstructured_scheme import (
    MeshToGridESMFRegridder,
)
from iris.cube import CubeList

from ugants.io.load import cf, ugrid
from ugants.regrid.applications import MeshToGridRegrid
from ugants.tests import get_data_path

OUTPUT_PATH = "/path/to/output.nc"


@pytest.fixture()
def source_path():
    return get_data_path("data_C4.nc")


@pytest.fixture()
def ugrid_cubelist(source_path):
    return ugrid(source_path)


@pytest.fixture()
def target_path():
    return get_data_path("non_ugrid_data.nc")


@pytest.fixture()
def regular_cubelist(target_path):
    return cf(target_path)


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

    def test_target_cube(self, default_app, regular_cubelist):
        assert default_app.target_grid == regular_cubelist

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


class TestRegrid:
    @pytest.fixture()
    def default_app(self, ugrid_cubelist, regular_cubelist):
        return MeshToGridRegrid(
            source=ugrid_cubelist[0:1],
            target_grid=regular_cubelist,
            horizontal_regrid_scheme="conservative",
        )

    def test_regridder_call(self, default_app, ugrid_cubelist):
        with mock.patch.object(
            MeshToGridESMFRegridder, "__call__"
        ) as mock_regridder_call:
            default_app.run()
        mock_regridder_call.assert_called_once_with(ugrid_cubelist[0])

    def test_multi_source_fail(self, ugrid_cubelist, regular_cubelist):
        sources = CubeList([ugrid_cubelist[0], ugrid_cubelist[0] + 1])
        app = MeshToGridRegrid(
            source=sources,
            target_grid=regular_cubelist,
            horizontal_regrid_scheme="conservative",
        )
        with pytest.raises(ValueError, match="Source contained 2 cubes, expected 1."):
            app.run()

    def test_multiple_target_fail(self, ugrid_cubelist, regular_cubelist):
        targets = CubeList([regular_cubelist[0], regular_cubelist[0] + 1])
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
