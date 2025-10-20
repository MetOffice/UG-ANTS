# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris.experimental.ugrid.mesh
from numpy.testing import assert_array_equal
from ugants.io import load
from ugants.tests import get_data_path
from ugants.utils.cube import get_connectivity_indices


class TestStartIndexOne:
    """Tests that 1 is subtracted off the connectivity array when start_index=1."""

    def test_face_face(self):
        """Test for face_face_connectivity."""
        source = load.ugrid(get_data_path("data_C4.nc")).extract_cube("sample_data")
        assert source.mesh.face_face_connectivity.start_index == 1
        raw_indices = source.mesh.face_face_connectivity.indices
        expected_indices = raw_indices - 1
        actual_indices = get_connectivity_indices(source, "face_face_connectivity")
        assert_array_equal(actual_indices, expected_indices)

    def test_face_node(self):
        """Test for face_node_connectivity."""
        source = load.ugrid(get_data_path("data_C4.nc")).extract_cube("sample_data")
        assert source.mesh.face_node_connectivity.start_index == 1
        raw_indices = source.mesh.face_node_connectivity.indices
        expected_indices = raw_indices - 1
        actual_indices = get_connectivity_indices(source, "face_node_connectivity")
        assert_array_equal(actual_indices, expected_indices)


class TestStartIndexZero:
    """Tests that 1 is not subtracted off the connectivity array when start_index=0."""

    def test_face_face(self):
        """Test for face_face_connectivity."""
        # data_C4.nc is one indexed, so we need to subtract 1 from all the
        # indices to create the expected indices for the zero indexed data.
        source = load.ugrid(get_data_path("data_C4.nc")).extract_cube("sample_data")
        assert source.mesh.face_face_connectivity.start_index == 1
        expected_indices = source.mesh.face_face_connectivity.indices - 1

        zero_indexed = _face_face_zero_index(source)
        assert zero_indexed.mesh.face_face_connectivity.start_index == 0
        actual_indices = get_connectivity_indices(
            zero_indexed, "face_face_connectivity"
        )

        assert_array_equal(actual_indices, expected_indices)


def _face_face_zero_index(cube):
    """Replace the face to face connectivity with a zero indexed equivalent."""
    cube = cube.copy()
    old_face_face = cube.mesh.face_face_connectivity
    new_face_face = iris.experimental.ugrid.mesh.Connectivity(
        indices=old_face_face.indices - 1, cf_role=old_face_face.cf_role, start_index=0
    )
    cube.mesh.remove_connectivities(old_face_face)
    cube.mesh.add_connectivities(new_face_face)
    return cube
