# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from unittest import mock

import pytest
from iris.experimental.ugrid import Mesh

from ugants.io.load import meshes
from ugants.tests import get_data_path


class TestFromFile:
    def test_mesh_no_name(self):
        """Test that a list of one mesh is returned when no mesh_name is provided."""
        filepath = get_data_path("mesh_C12.nc")
        mesh_list = meshes(filepath)
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 1
        mesh = mesh_list[0]
        assert isinstance(mesh, Mesh)
        assert mesh.var_name == "dynamics"

    def test_mesh_correct_name(self):
        """Test that a list of one mesh is returned when the mesh_name is provided."""
        filepath = get_data_path("mesh_C12.nc")
        mesh_list = meshes(filepath, "dynamics")
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 1
        mesh = mesh_list[0]
        assert isinstance(mesh, Mesh)
        assert mesh.var_name == "dynamics"

    def test_mesh_incorrect_name_fails(self):
        """Test that an error is raised when the wrong mesh_name is provided."""
        filepath = get_data_path("mesh_C12.nc")
        with pytest.raises(
            ValueError,
            match=f"No meshes were found at {filepath} with var_name 'physics'",
        ):
            meshes(filepath, "physics")

    def test_mesh_from_data_no_name(self):
        """Test that a list of one mesh is loaded from a UGrid data file."""
        filepath = get_data_path("data_C4.nc")
        mesh_list = meshes(filepath)
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 1
        mesh = mesh_list[0]
        assert isinstance(mesh, Mesh)
        assert mesh.var_name == "dynamics"

    def test_mesh_from_data_correct_name(self):
        """Test that a list of one mesh is loaded from a UGrid data file."""
        filepath = get_data_path("data_C4.nc")
        mesh_list = meshes(filepath, "dynamics")
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 1
        mesh = mesh_list[0]
        assert isinstance(mesh, Mesh)
        assert mesh.var_name == "dynamics"

    def test_mesh_from_data_incorrect_name_fails(self):
        """Test that an error is raised when the wrong mesh name is provided."""
        filepath = get_data_path("data_C4.nc")
        with pytest.raises(
            ValueError,
            match=f"No meshes were found at {filepath} with var_name 'physics'",
        ):
            meshes(filepath, "physics")

    def test_multi_mesh_no_name(self):
        """Test that all meshes are loaded when no name is provided."""
        filepath = get_data_path("multi_mesh_C24.nc")
        mesh_list = meshes(filepath)
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 2

        expected_names = {"dynamics", "physics"}
        actual_names = {mesh.var_name for mesh in mesh_list}
        assert actual_names == expected_names

    def test_multi_filepath_no_name(self):
        """Test that all meshes are loaded when no name is provided."""
        filepaths = [get_data_path("mesh_C12.nc"), get_data_path("multi_mesh_C24.nc")]
        mesh_list = meshes(filepaths)
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 3

        expected_names = {"dynamics", "physics"}
        actual_names = {mesh.var_name for mesh in mesh_list}
        assert actual_names == expected_names

    def test_multi_filepath_with_name(self):
        """Test that corect meshes are loaded when a name is provided."""
        filepaths = [get_data_path("mesh_C12.nc"), get_data_path("multi_mesh_C24.nc")]
        mesh_list = meshes(filepaths, "dynamics")
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 2

        assert all(mesh.var_name == "dynamics" for mesh in mesh_list)

    @pytest.mark.parametrize("mesh_name", ["dynamics", "physics"])
    def test_multi_mesh_with_name(self, mesh_name):
        """Test that the correct mesh is loaded when a name is provided."""
        filepath = get_data_path("multi_mesh_C24.nc")
        mesh_list = meshes(filepath, mesh_name)
        assert isinstance(mesh_list, list)
        assert len(mesh_list) == 1
        mesh = mesh_list[0]
        assert mesh.var_name == mesh_name

    def test_multi_mesh_with_incorrect_name_fails(self):
        """Test that an error is raised when an incorrect name is provided."""
        filepath = get_data_path("multi_mesh_C24.nc")
        with pytest.raises(
            ValueError, match=f"No meshes were found at {filepath} with var_name 'foo'"
        ):
            meshes(filepath, "foo")

    def test_non_netcdf_fails(self):
        """Test that an error is raised when the provided file is not netCDF."""
        with pytest.raises(ValueError, match="Input file 'foo.pp' must be netCDF."):
            meshes("foo.pp")

    def test_non_ugrid_data_fails(self):
        """Test that an error is raised when the file contains non-ugrid data."""
        filepath = get_data_path("non_ugrid_data.nc")
        with pytest.raises(ValueError, match=f"No meshes were found at {filepath}"):
            meshes(filepath)


class TestUnpackDict:
    """Test that the dict returned from the iris load_meshes function is unpacked.

    Mock the iris.experimental.ugrid.load_meshes function and return a dummy dictionary
    which maps each uri to a list of "meshes". Use integers as placeholders for actual
    mesh objects in these tests, since we are only testing the unpacking logic.
    """

    patch_target = "ugants.io.load.load_meshes"

    def test_unpack_single_filepath_single_mesh(self):
        dummy_loaded = {"dummy/filepath.nc": [0]}
        expected_meshes = [0]

        with mock.patch(self.patch_target, return_value=dummy_loaded):
            actual_meshes = meshes("dummy/filepath.nc")

        assert actual_meshes == expected_meshes

    def test_unpack_single_filepath_multi_mesh(self):
        dummy_loaded = {"dummy/filepath.nc": [0, 1, 2]}
        expected_meshes = [0, 1, 2]

        with mock.patch(self.patch_target, return_value=dummy_loaded):
            actual_meshes = meshes("dummy/filepath.nc")

        assert actual_meshes == expected_meshes

    def test_unpack_multi_filepath_multi_mesh(self):
        dummy_loaded = {
            "dummy/filepath0.nc": [0, 1],
            "dummy/filepath1.nc": [5, 10, 15],
            "dummy/filepath2.nc": [3],
        }
        expected_meshes = [0, 1, 5, 10, 15, 3]
        dummy_filepaths = dummy_loaded.keys()
        with mock.patch(self.patch_target, return_value=dummy_loaded):
            actual_meshes = meshes(dummy_filepaths)

        assert actual_meshes == expected_meshes
