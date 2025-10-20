# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the Application ABC."""
import re
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
from typing import Literal
from unittest import mock

import pytest
from iris.cube import CubeList
from iris.experimental.ugrid import Mesh
from ugants.abc import Application
from ugants.io import load
from ugants.tests import get_data_path

MESH_NAME = "dynamics"
OUTPUT_PATH = "/path/to/output.nc"


@dataclass
class ConcreteApplication(Application):
    """Concrete example app inheriting from Application ABC."""

    source: CubeList
    number: int
    mesh: Mesh
    scheme: Literal["conservative", "bilinear"] = "conservative"

    def run(self):
        self.results = self.source


@pytest.fixture()
def source_path():
    return get_data_path("data_C4.nc")


@pytest.fixture()
def mesh_path():
    return get_data_path("mesh_C12.nc")


@pytest.fixture()
def source_cubelist(source_path):
    return load.ugrid(source_path)


@pytest.fixture()
def mesh_C12(mesh_path):
    return load.mesh(mesh_path, mesh_name=MESH_NAME)


@pytest.fixture()
def default_command(source_path, mesh_path):
    return [
        source_path,
        OUTPUT_PATH,
        "--number",
        "1",
        "--mesh",
        mesh_path,
        "--mesh-name",
        MESH_NAME,
    ]


class TestFromCommandLine:
    """Tests for the :meth:`ugants.abc.Application.from_command_line` method."""

    def test_help_message(self):
        expected_help = r"""^usage: .* \[-h\] --number NUMBER \[--mesh-name MESH_NAME\] --mesh MESH
 *\[--scheme {conservative,bilinear}\]
 *source output

Concrete example app inheriting from Application ABC.

positional arguments:
  source                Path to source.
  output                Path to output the results from ConcreteApplication.

options:
  -h, --help            show this help message and exit
  --number NUMBER
  --mesh-name MESH_NAME
                        The name of the mesh contained in the file provided to
                        '--mesh'.
  --mesh MESH           Path to mesh.
  --scheme {conservative,bilinear}
                        Default is conservative.
$"""  # noqa: E501

        actual_help = ConcreteApplication._parser().format_help()
        assert re.search(expected_help, actual_help)

    def test_source_loaded(self, default_command, source_cubelist):
        expected_source = source_cubelist
        app = ConcreteApplication.from_command_line(default_command)
        actual_source = app.source
        assert actual_source == expected_source

    def test_default_argument(self, default_command):
        app = ConcreteApplication.from_command_line(default_command)
        assert app.scheme == "conservative"

    def test_optional_argument(self, default_command):
        command = default_command
        command.extend(["--scheme", "bilinear"])
        app = ConcreteApplication.from_command_line(command)
        assert app.scheme == "bilinear"

    def test_invalid_optional_argument(self, default_command):
        command = default_command
        command.extend(["--scheme", "nearest"])
        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            ConcreteApplication.from_command_line(command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile(
            "error: argument --scheme: invalid choice: 'nearest' "
            r"\(choose from 'conservative', 'bilinear'\)"
        )
        assert expected_stderr.search(actual_stderr)

    def test_output_added(self, default_command):
        app = ConcreteApplication.from_command_line(default_command)
        assert app.output == OUTPUT_PATH

    def test_mesh_loaded(self, default_command, mesh_C12):
        app = ConcreteApplication.from_command_line(default_command)
        assert app.mesh == mesh_C12

    def test_no_mesh_name_given(self, source_path, mesh_path, mesh_C12):
        """Tests that the mesh will be loaded when not supplied with a mesh name."""
        command = [
            source_path,
            OUTPUT_PATH,
            "--number",
            "1",
            "--mesh",
            mesh_path,
        ]
        app = ConcreteApplication.from_command_line(command)
        assert app.mesh == mesh_C12

    def test_abbreviated_required_argument_fails(self, default_command):
        # Replace --number with --num
        abbreviated_command = default_command
        abbreviated_command[2] = "--num"
        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            ConcreteApplication.from_command_line(abbreviated_command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile(
            "error: the following arguments are required: --number"
        )
        assert expected_stderr.search(actual_stderr)

    def test_abbreviated_optional_argument_fails(self, default_command):
        # --scheme is optional, so check that "--sch bilinear" doesn't silently
        # pass the default argument "conservative": it should fail
        abbreviated_command = default_command
        abbreviated_command.extend(["--sch", "bilinear"])
        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            ConcreteApplication.from_command_line(abbreviated_command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile("error: unrecognized arguments: --sch bilinear")
        assert expected_stderr.search(actual_stderr)


class TestParser:
    def test_arguments_added(self):
        add_argument = "ugants.abc.argparse.ArgumentParser.add_argument"
        with mock.patch(add_argument) as mock_add_args:
            ConcreteApplication._parser()

        mock_add_args.assert_has_calls(
            [
                mock.call("source", help="Path to source.", type=str),
                mock.call(
                    "output",
                    help="Path to output the results from ConcreteApplication.",
                ),
                mock.call("--number", help="", type=int, required=True),
                mock.call("--mesh", help="Path to mesh.", type=str, required=True),
                mock.call(
                    "--mesh-name",
                    help="The name of the mesh contained in the file "
                    "provided to '--mesh'.",
                    type=str,
                    required=False,
                ),
                mock.call(
                    "--scheme",
                    help="Default is conservative.",
                    choices=("conservative", "bilinear"),
                    default="conservative",
                    required=False,
                    type=None,
                ),
            ],
            any_order=True,
        )

    def test_description_added(self):
        parser = ConcreteApplication._parser()
        actual_description = parser.description
        expected_description = ConcreteApplication.__doc__
        assert actual_description == expected_description


class NoAbstractMethods(Application):
    pass


class NoInit(Application):
    def run(self):
        pass


class NoRun(Application):
    def __init__(self):
        pass


class TestAbstractMethods:
    def test_no_init_or_run_fails(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class NoAbstractMethods with abstract "
            "methods __init__, run",
        ):
            NoAbstractMethods()

    def test_no_init_fails(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class NoInit with abstract "
            "method __init__",
        ):
            NoInit()

    def test_no_run_fails(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class NoRun with abstract method run",
        ):
            NoRun()


class TestSave:
    def test_save_with_output_added(self, source_cubelist, mesh_C12):
        app = ConcreteApplication(source=source_cubelist, number=1, mesh=mesh_C12)
        app.run()
        app.output = OUTPUT_PATH
        with mock.patch("ugants.abc.save.ugrid") as mock_save:
            app.save()
        mock_save.assert_called_once_with(app.results, OUTPUT_PATH)

    def test_save_with_output_from_command_line(self, default_command):
        app = ConcreteApplication.from_command_line(default_command)
        app.run()
        with mock.patch("ugants.abc.save.ugrid") as mock_save:
            app.save()
        mock_save.assert_called_once_with(app.results, OUTPUT_PATH)

    def test_save_with_no_output_fails(self, source_cubelist, mesh_C12):
        app = ConcreteApplication(source=source_cubelist, number=1, mesh=mesh_C12)
        app.run()
        with pytest.raises(ValueError, match="No output file location has been set."):
            app.save()

    def test_save_with_no_result_fails(self, source_cubelist, mesh_C12):
        app = ConcreteApplication(source=source_cubelist, number=1, mesh=mesh_C12)
        app.output = OUTPUT_PATH
        with pytest.raises(
            ValueError, match="The application has not yet been run, results is None."
        ):
            app.save()
