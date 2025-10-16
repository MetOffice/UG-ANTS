# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import pytest
from iris.experimental.ugrid.mesh import Mesh

from ugants.io.load import mesh
from ugants.tests import get_data_path


class TestLoadMesh:
    def test_load_single_mesh(self):
        input_filepath = get_data_path("mesh_C12.nc")
        target_mesh = mesh(input_filepath, "dynamics")
        assert isinstance(target_mesh, Mesh) is True
        assert target_mesh.var_name == "dynamics"

    def test_load_single_mesh_from_multi(self):
        input_filepath = get_data_path("multi_mesh_C24.nc")
        target_mesh = mesh(input_filepath, "physics")
        assert isinstance(target_mesh, Mesh) is True
        assert target_mesh.var_name == "physics"

    def test_load_mesh_with_data(self):
        input_filepath = get_data_path("data_C4.nc")
        target_mesh = mesh(input_filepath, "dynamics")
        assert isinstance(target_mesh, Mesh) is True
        assert target_mesh.var_name == "dynamics"

    def test_load_regular_data(self):
        input_filepath = get_data_path("non_ugrid_data.nc")
        error_message = (
            f"No meshes were found at {input_filepath} with var_name 'dynamics'"
        )
        with pytest.raises(ValueError) as excinfo:
            mesh(input_filepath, "dynamics")

        assert str(excinfo.value) == error_message

    def test_load_single_mesh_no_name(self):
        input_filepath = get_data_path("mesh_C12.nc")
        target_mesh = mesh(input_filepath)
        assert target_mesh.var_name == "dynamics"

    def test_load_multiple_mesh_no_name(self):
        input_filepath = get_data_path("multi_mesh_C24.nc")
        error_message = "Expected one mesh, found 2. ['dynamics', 'physics']"
        with pytest.raises(ValueError) as excinfo:
            mesh(input_filepath)

        assert str(excinfo.value) == error_message
