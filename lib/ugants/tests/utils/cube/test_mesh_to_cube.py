# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest
from iris.cube import Cube
from ugants.tests.stock import cubedsphere_mesh
from ugants.utils.cube import mesh_to_cube


@pytest.fixture()
def c12_mesh():
    mesh = cubedsphere_mesh(12)
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
        assert np.all(cube.data.mask)

    def test_fill_custom_dtype(self, c12_mesh):
        dtype = np.float32
        cube = mesh_to_cube(c12_mesh, dtype=dtype)
        assert cube.data.dtype == np.float32

    def test_custom_data(self, c12_mesh):
        custom_data = np.ma.arange(12 * 12 * 6)
        cube = mesh_to_cube(c12_mesh, data=custom_data)
        np.testing.assert_array_equal(cube.data, custom_data)
