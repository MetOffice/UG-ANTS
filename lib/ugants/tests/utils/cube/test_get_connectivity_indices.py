# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris.experimental.ugrid.mesh
from numpy.testing import assert_array_equal
from ugants.tests.stock import cubedsphere_cube
from ugants.utils.cube import get_connectivity_indices


class TestStartIndexOne:
    """Tests that 1 is subtracted off the connectivity array when start_index=1."""

    def test_face_face(self):
        """Test for face_face_connectivity.

        Create mesh with start index 1, then check that 1 is subtracted off.
        """
        source = cubedsphere_cube(4)
        source = _reindex_zero_to_one(source, "face_face_connectivity")
        assert source.mesh.face_face_connectivity.start_index == 1

        raw_indices = source.mesh.face_face_connectivity.indices
        expected_indices = raw_indices - 1
        actual_indices = get_connectivity_indices(source, "face_face_connectivity")
        assert_array_equal(actual_indices, expected_indices)

    def test_face_node(self):
        """Test for face_node_connectivity.

        Create mesh with start index 1, then check that 1 is subtracted off.
        """
        source = cubedsphere_cube(4)
        source = _reindex_zero_to_one(source, "face_node_connectivity")
        assert source.mesh.face_node_connectivity.start_index == 1
        raw_indices = source.mesh.face_node_connectivity.indices
        expected_indices = raw_indices - 1
        actual_indices = get_connectivity_indices(source, "face_node_connectivity")
        assert_array_equal(actual_indices, expected_indices)


class TestStartIndexZero:
    """Tests that 1 is not subtracted off the connectivity array when start_index=0."""

    def test_face_face(self):
        """Test for face_face_connectivity."""
        source = cubedsphere_cube(4)
        assert source.mesh.face_face_connectivity.start_index == 0
        expected_indices = source.mesh.face_face_connectivity.indices

        actual_indices = get_connectivity_indices(source, "face_face_connectivity")

        assert_array_equal(actual_indices, expected_indices)

    def test_face_node(self):
        """Test for face_node_connectivity."""
        source = cubedsphere_cube(4)
        assert source.mesh.face_node_connectivity.start_index == 0
        expected_indices = source.mesh.face_node_connectivity.indices

        actual_indices = get_connectivity_indices(source, "face_node_connectivity")

        assert_array_equal(actual_indices, expected_indices)


def _reindex_zero_to_one(cube, cf_role):
    """Replace a 0 indexed connectivity with a 1 indexed equivalent."""
    cube = cube.copy()
    old_connectivity = cube.mesh.connectivity(cf_role=cf_role)
    assert old_connectivity.start_index == 0
    new_connectivity = iris.experimental.ugrid.mesh.Connectivity(
        indices=old_connectivity.indices + 1, cf_role=cf_role, start_index=1
    )
    cube.mesh.remove_connectivities(old_connectivity)
    cube.mesh.add_connectivities(new_connectivity)
    return cube
