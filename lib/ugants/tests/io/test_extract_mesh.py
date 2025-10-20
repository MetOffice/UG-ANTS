# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the :class:`ugants.io.applications.ExtractSingleMesh` function."""
from unittest import mock

import pytest

from ugants.io import load
from ugants.io.applications import ExtractSingleMesh
from ugants.tests import get_data_path


@pytest.fixture()
def multi_mesh_path():
    """Return a path to a file containing two C24 meshes: dynamics and physics."""
    return get_data_path("multi_mesh_C24.nc")


@pytest.fixture()
def single_mesh(multi_mesh_path):
    """Return a list containing a single C24 mesh: dynamics."""
    return load.mesh(multi_mesh_path, "dynamics")


class TestLoad:
    def test_mesh_loaded(self, multi_mesh_path, single_mesh):
        command = ["--mesh", multi_mesh_path, "--mesh-name", "dynamics", "output"]
        app = ExtractSingleMesh.from_command_line(command)
        assert single_mesh == app.mesh

    def test_load_call(self, multi_mesh_path):
        command = ["--mesh", multi_mesh_path, "--mesh-name", "physics", "output"]
        patch_target = "ugants.abc.load.mesh"
        with mock.patch(patch_target) as mock_load:
            ExtractSingleMesh.from_command_line(command)
        mock_load.assert_called_once_with(multi_mesh_path, "physics")


class TestSave:
    def test_save_call(self, multi_mesh_path, single_mesh):
        command = ["--mesh", multi_mesh_path, "--mesh-name", "dynamics", "output"]
        app = ExtractSingleMesh.from_command_line(command)
        patch_target = "ugants.io.applications.save.mesh"
        with mock.patch(patch_target) as mock_save:
            app.save()
        mock_save.assert_called_once_with(single_mesh, "output")


class TestRun:
    def test_error_on_run(self, single_mesh):
        """The run method is not implemented, should raise a RuntimeError."""
        app = ExtractSingleMesh(single_mesh)
        with pytest.raises(
            RuntimeError,
            match="^The run method is not implemented for this application.$",
        ):
            app.run()
