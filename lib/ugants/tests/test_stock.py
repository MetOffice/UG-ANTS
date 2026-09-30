# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office GitHub Copilot Enterprise.
from abc import ABC, abstractmethod

import numpy as np
import pytest
import slam
from iris.cube import Cube
from iris.experimental.ugrid import Mesh

import ugants.tests.stock


class CommonMesh(ABC):
    """Base class for testing of stock meshes."""

    side_lengths = (4, 8)

    @abstractmethod
    def make_mesh(self, side_length: int) -> Mesh:
        """Build the mesh under test."""

    @abstractmethod
    def expected_n_faces(self, side_length: int) -> int:
        """Return the expected number of faces in the mesh."""

    @abstractmethod
    def expected_n_nodes(self, side_length: int) -> int:
        """Return the expected number of nodes in the mesh."""

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_returns_mesh(self, side_length):
        mesh = self.make_mesh(side_length)
        assert isinstance(mesh, Mesh)

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_face_node_connectivity_shape(self, side_length):
        mesh = self.make_mesh(side_length)
        indices = mesh.face_node_connectivity.indices
        assert indices.shape == (self.expected_n_faces(side_length), 4)

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_face_node_connectivity_is_masked_array(self, side_length):
        mesh = self.make_mesh(side_length)
        assert isinstance(mesh.face_node_connectivity.indices, np.ma.MaskedArray)

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_face_face_connectivity_shape(self, side_length):
        mesh = self.make_mesh(side_length)
        face_face = mesh.face_face_connectivity
        assert face_face.indices.shape == (self.expected_n_faces(side_length), 4)

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_face_coords(self, side_length):
        mesh = self.make_mesh(side_length)
        assert mesh.face_coords.face_x is not None
        assert mesh.face_coords.face_y is not None
        assert mesh.face_coords.face_x.points.shape == (
            self.expected_n_faces(side_length),
        )
        assert mesh.face_coords.face_y.points.shape == (
            self.expected_n_faces(side_length),
        )

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_node_coords(self, side_length):
        mesh = self.make_mesh(side_length)
        assert mesh.node_coords.node_x is not None
        assert mesh.node_coords.node_y is not None
        assert mesh.node_coords.node_x.points.shape == (
            self.expected_n_nodes(side_length),
        )
        assert mesh.node_coords.node_y.points.shape == (
            self.expected_n_nodes(side_length),
        )


class TestPanelMesh(CommonMesh):
    """Tests for ugants.tests.stock.panel_mesh."""

    def make_mesh(self, side_length: int) -> Mesh:
        return ugants.tests.stock.panel_mesh(side_length)

    def expected_n_faces(self, side_length: int) -> int:
        return side_length**2

    def expected_n_nodes(self, side_length: int) -> int:
        return (side_length + 1) ** 2

    def test_raises_for_zero_side_length(self):
        expected_msg = r"^Panel side length must be positive \(requested 0\)$"
        with pytest.raises(ValueError, match=expected_msg):
            ugants.tests.stock.panel_mesh(0)

    def test_raises_for_side_length_above_12(self):
        expected_msg = (
            r"^The panel mesh generator does not support "
            r"side lengths above 12 \(requested 13\)$"
        )
        with pytest.raises(ValueError, match=expected_msg):
            ugants.tests.stock.panel_mesh(13)


class CommonCube(ABC):
    """Base class for testing of stock cubes."""

    side_lengths = (4, 8)

    @abstractmethod
    def make_cube(self, side_length: int) -> Cube:
        """Build the cube under test."""

    @abstractmethod
    def expected_shape(self, side_length: int) -> tuple[int, ...]:
        """Return the expected shape of the cube data."""

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_returns_cube(self, side_length):
        cube = self.make_cube(side_length)
        assert isinstance(cube, Cube)

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_shape(self, side_length):
        cube = self.make_cube(side_length)
        assert cube.shape == self.expected_shape(side_length)

    @pytest.mark.parametrize("side_length", side_lengths)
    def test_has_mesh(self, side_length):
        cube = self.make_cube(side_length)
        assert cube.mesh is not None
        assert isinstance(cube.mesh, Mesh)


class TestPanelCube(CommonCube):
    """Tests for ugants.tests.stock.panel_cube."""

    def make_cube(self, side_length: int) -> Cube:
        return ugants.tests.stock.panel_cube(side_length)

    def expected_shape(self, side_length: int) -> tuple[int, ...]:
        return (side_length**2,)

    @pytest.mark.parametrize("side_length", [4, 8])
    def test_slam_from_ugrid(self, side_length):
        cube = ugants.tests.stock.panel_cube(side_length)
        result = slam.Transform.from_ugrid(cube)
        assert isinstance(result, Cube)
        assert result.shape == (side_length, side_length)

    @pytest.mark.parametrize("side_length", [4, 8])
    def test_slam_from_ugrid_has_no_mesh(self, side_length):
        cube = ugants.tests.stock.panel_cube(side_length)
        result = slam.Transform.from_ugrid(cube)
        assert result.mesh is None


class TestCubedsphereMesh(CommonMesh):
    """Tests for ugants.tests.stock.cubedsphere_mesh."""

    def make_mesh(self, side_length: int) -> Mesh:
        return ugants.tests.stock.cubedsphere_mesh(side_length)

    def expected_n_faces(self, side_length: int) -> int:
        return 6 * side_length**2

    def expected_n_nodes(self, side_length: int) -> int:
        return 6 * side_length**2 + 2

    def test_raises_for_zero_side_length(self):
        expected_msg = r"^Cubedsphere side length must be positive \(requested 0\)$"
        with pytest.raises(ValueError, match=expected_msg):
            ugants.tests.stock.cubedsphere_mesh(0)

    def test_raises_for_side_length_above_12(self):
        expected_msg = (
            r"^The cubedsphere mesh generator does not support "
            r"side lengths above 12 \(requested 13\)$"
        )
        with pytest.raises(ValueError, match=expected_msg):
            ugants.tests.stock.cubedsphere_mesh(13)


class TestCubedsphereCube(CommonCube):
    """Tests for ugants.tests.stock.cubedsphere_cube."""

    def make_cube(self, side_length: int) -> Cube:
        return ugants.tests.stock.cubedsphere_cube(side_length)

    def expected_shape(self, side_length: int) -> tuple[int, ...]:
        return (6 * side_length**2,)

    @pytest.mark.parametrize("side_length", [1, 2, 4])
    def test_panel_number_values(self, side_length):
        cube = ugants.tests.stock.cubedsphere_cube(side_length)
        panel_number = cube.coord("panel_number").points
        expected = np.repeat(np.arange(6), side_length**2)
        np.testing.assert_array_equal(panel_number, expected)


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

    mesh = ugants.tests.stock.regular_lat_lon_mesh(0, 15, 0, 10, (3, 2))
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

    mesh = ugants.tests.stock.regular_lat_lon_mesh(0, 10, 0, 15, (2, 3))
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
