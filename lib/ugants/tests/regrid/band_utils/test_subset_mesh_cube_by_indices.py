# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from unittest import mock

import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from ugants.io import load
from ugants.regrid.band_utils import subset_mesh_cube_by_indices
from ugants.tests import get_data_path


@pytest.fixture()
def sample_mesh_cube():
    return load.ugrid(get_data_path("data_C4.nc")).extract_cube("sample_data")


class TestExceptions:
    def test_no_mesh(self):
        """Test that an error is raised if the cube has no mesh."""
        cube = Cube([0])
        assert cube.mesh is None
        with pytest.raises(
            ValueError,
            match="The provided cube does not have a mesh, expected one to be present.",
        ):
            subset_mesh_cube_by_indices(cube, [0, 2])

    @pytest.mark.parametrize("location", ["node", "edge", None])
    def test_non_face_location(self, location):
        cube = mock.Mock(spec=Cube)
        cube.location = location
        expected_message = (
            "Mesh subsetting is currently only available for data located "
            f"on faces, the provided cube has location {location}."
        )

        with pytest.raises(ValueError, match=expected_message):
            subset_mesh_cube_by_indices(cube, [0, 2])


class TestReconstructMeshCallArgs:
    def test_1D_mesh_cube(self, sample_mesh_cube):
        """Test cube with one unstructured dimension."""
        assert sample_mesh_cube.mesh_dim() == 0

        indices = [0, 2]
        expected_subset_cube = sample_mesh_cube[indices]

        with mock.patch(
            "ugants.regrid.band_utils.reconstruct_mesh_cube"
        ) as mock_reconstruct:
            subset_mesh_cube_by_indices(sample_mesh_cube, indices)
        mock_reconstruct.assert_called_once_with(expected_subset_cube, 0)

    def test_2D_mesh_cube(self, sample_mesh_cube):
        """Test cube with one unstructured and one structured dimension."""
        time = AuxCoord([1, 2], standard_name="time")
        mesh_cube_3x3_time1 = sample_mesh_cube.copy()
        mesh_cube_3x3_time1.add_aux_coord(time[0])
        mesh_cube_3x3_time2 = sample_mesh_cube.copy()
        mesh_cube_3x3_time2.add_aux_coord(time[1])
        multi_dim_cube = CubeList([mesh_cube_3x3_time1, mesh_cube_3x3_time2]).merge()[0]
        assert multi_dim_cube.ndim == 2
        assert multi_dim_cube.mesh_dim() == 1

        indices = [0, 2]
        expected_subset_cube = multi_dim_cube[:, indices]

        with mock.patch(
            "ugants.regrid.band_utils.reconstruct_mesh_cube"
        ) as mock_reconstruct:
            subset_mesh_cube_by_indices(multi_dim_cube, indices)
        mock_reconstruct.assert_called_once_with(expected_subset_cube, 1)


class TestMetadata:
    """Tests that metadata is preserved under mesh subsetting."""

    def test_mesh_names(self, sample_mesh_cube):
        expected_var_name = sample_mesh_cube.mesh.var_name
        expected_standard_name = sample_mesh_cube.mesh.standard_name
        expected_long_name = sample_mesh_cube.mesh.long_name

        subset_cube = subset_mesh_cube_by_indices(sample_mesh_cube, [0, 2])

        assert subset_cube.mesh.var_name == expected_var_name
        assert subset_cube.mesh.standard_name == expected_standard_name
        assert subset_cube.mesh.long_name == expected_long_name

    def test_mesh_dimensions(self, sample_mesh_cube):
        expected_node_dimension = sample_mesh_cube.mesh.node_dimension
        expected_edge_dimension = sample_mesh_cube.mesh.edge_dimension
        expected_face_dimension = sample_mesh_cube.mesh.face_dimension

        subset_cube = subset_mesh_cube_by_indices(sample_mesh_cube, [0, 2])

        assert subset_cube.mesh.node_dimension == expected_node_dimension
        assert subset_cube.mesh.edge_dimension == expected_edge_dimension
        assert subset_cube.mesh.face_dimension == expected_face_dimension

    def test_mesh_units(self, sample_mesh_cube):
        expected_units = sample_mesh_cube.mesh.units

        subset_cube = subset_mesh_cube_by_indices(sample_mesh_cube, [0, 2])

        assert subset_cube.mesh.units == expected_units
