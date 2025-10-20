# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest

from ugants.tests.stock import regular_lat_lon_mesh


class TestDummyRegularMesh3x2:
    """Test the faces and nodes for a 3 by 2 dummy mesh.

    10.0  [2]-------[5]-------[8]-------[11]
           |         |         |         |
    7.5    |   (1)   |   (3)   |   (5)   |
           |         |         |         |
    5.0   [1]-------[4]-------[7]-------[10]
           |         |         |         |
    2.5    |   (0)   |   (2)   |   (4)   |
           |         |         |         |
    0.0   [0]-------[3]-------[6]-------[9]

          0.0  2.5  5.0  7.5 10.0 12.5 15.0
    """

    mesh = regular_lat_lon_mesh(0, 15, 0, 10, (3, 2))
    face_x_points = mesh.face_coords.face_x.points
    face_y_points = mesh.face_coords.face_y.points
    node_x_points = mesh.node_coords.node_x.points
    node_y_points = mesh.node_coords.node_y.points
    face_node_indices = mesh.face_node_connectivity.indices

    def test_face_centres(self):
        expected_x = np.array([2.5, 2.5, 7.5, 7.5, 12.5, 12.5])
        expected_y = np.array([2.5, 7.5, 2.5, 7.5, 2.5, 7.5])
        np.testing.assert_array_equal(self.face_x_points, expected_x)
        np.testing.assert_array_equal(self.face_y_points, expected_y)

    def test_node_coords(self):
        expected_x = np.array([0, 0, 0, 5, 5, 5, 10, 10, 10, 15, 15, 15])
        expected_y = np.array([0, 5, 10, 0, 5, 10, 0, 5, 10, 0, 5, 10])
        np.testing.assert_array_equal(self.node_x_points, expected_x)
        np.testing.assert_array_equal(self.node_y_points, expected_y)

    expected_face_node_connectivity = (
        {0, 1, 3, 4},
        {1, 2, 4, 5},
        {3, 4, 6, 7},
        {4, 5, 7, 8},
        {6, 7, 9, 10},
        {7, 8, 10, 11},
    )

    @pytest.mark.parametrize(
        ("face_index", "expected_nodes"), enumerate(expected_face_node_connectivity)
    )
    def test_face_node_connectivity(self, face_index, expected_nodes):
        actual_connected_nodes = set(self.face_node_indices[face_index])
        assert actual_connected_nodes == expected_nodes

    def test_face_node_connectivity_is_masked_array(self):
        assert isinstance(self.face_node_indices, np.ma.MaskedArray)


class TestDummyRegularMesh2x3:
    """Test the faces and nodes for a 2 by 3 dummy mesh.

    15.0  [3]-------[7]-------[11]
           |         |         |
    12.5   |   (2)   |   (5)   |
           |         |         |
    10.0  [2]-------[6]-------[10]
           |         |         |
    7.5    |   (1)   |   (4)   |
           |         |         |
    5.0   [1]-------[5]-------[9]
           |         |         |
    2.5    |   (0)   |   (3)   |
           |         |         |
    0.0   [0]-------[4]-------[8]

          0.0  2.5  5.0  7.5  10.0
    """

    mesh = regular_lat_lon_mesh(0, 10, 0, 15, (2, 3))
    face_x_points = mesh.face_coords.face_x.points
    face_y_points = mesh.face_coords.face_y.points
    node_x_points = mesh.node_coords.node_x.points
    node_y_points = mesh.node_coords.node_y.points
    face_node_indices = mesh.face_node_connectivity.indices

    def test_face_centres(self):
        expected_x = np.array([2.5, 2.5, 2.5, 7.5, 7.5, 7.5])
        expected_y = np.array([2.5, 7.5, 12.5, 2.5, 7.5, 12.5])
        np.testing.assert_array_equal(expected_x, self.face_x_points)
        np.testing.assert_array_equal(expected_y, self.face_y_points)

    def test_node_coords(self):
        expected_x = np.array([0, 0, 0, 0, 5, 5, 5, 5, 10, 10, 10, 10])
        expected_y = np.array([0, 5, 10, 15, 0, 5, 10, 15, 0, 5, 10, 15])
        np.testing.assert_array_equal(self.node_x_points, expected_x)
        np.testing.assert_array_equal(self.node_y_points, expected_y)

    expected_face_node_connectivity = (
        {0, 1, 4, 5},
        {1, 2, 5, 6},
        {2, 3, 6, 7},
        {4, 5, 8, 9},
        {5, 6, 9, 10},
        {6, 7, 10, 11},
    )

    @pytest.mark.parametrize(
        ("face_index", "expected_nodes"), enumerate(expected_face_node_connectivity)
    )
    def test_face_node_connectivity(self, face_index, expected_nodes):
        actual_connected_nodes = set(self.face_node_indices[face_index])
        assert actual_connected_nodes == expected_nodes
