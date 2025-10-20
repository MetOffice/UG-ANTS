# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest
from iris.cube import Cube

from ugants.io.load import mesh as load_mesh
from ugants.regrid.band_utils import (
    mesh_to_cube,
)
from ugants.tests import get_data_path


@pytest.fixture()
def c12_mesh():
    input_filepath = get_data_path("mesh_C12.nc")
    mesh = load_mesh(input_filepath, "dynamics")
    return mesh


class TestMeshToCube:
    def test_equivalent_mesh(self, c12_mesh):
        cube = mesh_to_cube(c12_mesh)
        assert isinstance(cube, Cube)
        assert c12_mesh == cube.mesh

    def test_fill_default_type(self, c12_mesh):
        dtype = np.float64
        cube = mesh_to_cube(c12_mesh)
        assert cube.data.dtype == dtype

    def test_fill_default_value(self, c12_mesh):
        cube = mesh_to_cube(c12_mesh)
        assert all(np.isnan(cube.data))

    def test_fill_custom_dtype(self, c12_mesh):
        dtype = np.float32
        cube = mesh_to_cube(c12_mesh, dtype=dtype)
        assert cube.data.dtype == np.float32
