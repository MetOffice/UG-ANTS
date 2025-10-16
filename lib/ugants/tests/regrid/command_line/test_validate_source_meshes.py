# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris.cube
import pytest

from ugants.regrid.command_line import _validate_source_meshes
from ugants.tests.stock import mesh_cube


class TestAll:
    def test_equal_meshes_no_error(self):
        cubes = iris.cube.CubeList([mesh_cube(n_faces=10), mesh_cube(n_faces=10)])
        assert _validate_source_meshes(cubes) is None

    def test_different_meshes_errors(self):
        cubes = iris.cube.CubeList([mesh_cube(n_faces=10), mesh_cube(n_faces=11)])
        with pytest.raises(
            ValueError, match="Not all source cubes have the same horizontal mesh."
        ):
            _validate_source_meshes(cubes)
