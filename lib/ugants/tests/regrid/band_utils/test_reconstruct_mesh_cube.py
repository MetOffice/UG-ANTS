# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import iris
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import CubeList

from ugants.regrid.band_utils import (
    mesh_to_cube,
    reconstruct_mesh_cube,
)
from ugants.tests.stock import regular_lat_lon_mesh


@pytest.fixture()
def mesh_cube_3x3():
    """Create an initial cube with a mesh to be subset.

    lat
    20  [3]-------[7]-------[11]------[15]
         |         |         |         |
    15   |   (2)   |   (5)   |   (8)   |
         |         |         |         |
    10  [2]-------[6]-------[10]------[14]
         |         |         |         |
    5    |   (1)   |   (4)   |   (7)   |
         |         |         |         |
    0   [1]-------[5]-------[9]-------[13]
         |         |         |         |
    -5   |   (0)   |   (3)   |   (6)   |
         |         |         |         |
    -10 [0]-------[4]-------[8]-------[12]

         0    5    10   15   20   25   30  lon
    """
    mesh = regular_lat_lon_mesh(
        min_lat=-10, max_lat=20, min_lon=0, max_lon=30, shape=(3, 3)
    )
    mesh_cube = mesh_to_cube(mesh)
    mesh_cube.data = np.arange(mesh_cube.shape[0])
    return mesh_cube


def get_face_node_xy(mesh):
    """Return the x and y coordinates of the face-node connectivity."""
    face_node_indices = (
        mesh.face_node_connectivity.indices_by_location()
        - mesh.face_node_connectivity.start_index
    )
    face_node_x = mesh.node_coords.node_x.points[face_node_indices]
    face_node_y = mesh.node_coords.node_y.points[face_node_indices]
    return face_node_x, face_node_y


def test_fails_if_already_mesh(mesh_cube_3x3):
    assert mesh_cube_3x3.mesh is not None
    with pytest.raises(ValueError, match="The provided cube already has a mesh."):
        reconstruct_mesh_cube(mesh_cube_3x3, mesh_cube_3x3.mesh_dim())


class TestSubsetByIndices:
    """Subset the cells 0, 3, 4, 6 from the 3x3 mesh using index slicing.

    Expected subset:
    lat
    10            [n]-------[n]
                   |         |
    5              |   (2)   |
                   |         |
    0   [n]-------[n]-------[n]-------[n]
         |         |         |         |
    -5   |   (0)   |   (1)   |   (3)   |
         |         |         |         |
    -10 [n]-------[n]-------[n]-------[n]

         0    5    10   15   20   25   30  lon

    Node order is not guaranteed, but face order is.
    """

    indices_to_subset = (0, 3, 4, 6)

    def test_number_of_faces(self, mesh_cube_3x3):
        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_faces = 4
        assert len(subset_mesh.face_coords.face_x.points) == expected_faces
        assert len(subset_mesh.face_coords.face_y.points) == expected_faces

    def test_face_coords(self, mesh_cube_3x3):
        """Test that the mesh faces have been subset in the order provided."""
        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_face_x = [5, 15, 15, 25]
        expected_face_y = [-5, -5, 5, -5]
        actual_face_x = subset_mesh.face_coords.face_x.points
        actual_face_y = subset_mesh.face_coords.face_y.points

        np.testing.assert_array_equal(actual_face_x, expected_face_x)
        np.testing.assert_array_equal(actual_face_y, expected_face_y)

    def test_number_of_nodes(self, mesh_cube_3x3):
        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_nodes = 10
        assert len(subset_mesh.node_coords.node_x.points) == expected_nodes
        assert len(subset_mesh.node_coords.node_y.points) == expected_nodes

    def test_node_coords(self, mesh_cube_3x3):
        """Test that the node locations are correct, order is not guaranteed.

        Although the node ordering is not guaranteed, the node_x and node_y arrays
        should have a consistent order, i.e. the value at index i of node_x and node_y
        both refer to the same node.
        """
        expected_nodes_x = [0, 0, 10, 10, 10, 20, 20, 20, 30, 30]
        expected_nodes_y = [-10, 0, -10, 0, 10, -10, 0, 10, -10, 0]
        # {(0, -10), (0, 0), (10, -10), (10, 0), (10, 10), (20, -10),
        #  (20, 0), (20, 10),(30, -10), (30, 0)}
        expected_nodes_xy = set(zip(expected_nodes_x, expected_nodes_y, strict=True))

        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_nodes_x = subset_mesh.node_coords.node_x.points
        actual_nodes_y = subset_mesh.node_coords.node_y.points
        actual_nodes_xy = set(zip(actual_nodes_x, actual_nodes_y, strict=True))

        assert actual_nodes_xy == expected_nodes_xy

    def test_face_node_connectivity(self, mesh_cube_3x3):
        """Test that the face node connectivity has been remapped to the new nodes.

        The actual coordinates of the nodes connected to each face should not have
        changed by subsetting the mesh. Their indices will have changed, and this
        should be reflected in a new face-node connectivity.
        """
        original_face_node_x, original_face_node_y = get_face_node_xy(
            mesh_cube_3x3.mesh
        )
        expected_face_node_x = original_face_node_x[self.indices_to_subset,]
        expected_face_node_y = original_face_node_y[self.indices_to_subset,]

        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_face_node_x, actual_face_node_y = get_face_node_xy(subset_mesh)

        np.testing.assert_array_equal(actual_face_node_x, expected_face_node_x)
        np.testing.assert_array_equal(actual_face_node_y, expected_face_node_y)

    def test_data_preserved(self, mesh_cube_3x3):
        """Test that the data array is preserved by the mesh reconstruction."""
        expected_data = mesh_cube_3x3.data[self.indices_to_subset,].copy()

        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        actual_data = subset_cube.data

        np.testing.assert_array_equal(actual_data, expected_data)

    def test_mask_preserved(self, mesh_cube_3x3):
        """Test that the mask array is preserved by the mesh reconstruction."""
        mask = np.zeros_like(mesh_cube_3x3.data, dtype=bool)
        mask[(0, 2, 4),] = True
        mesh_cube_3x3.data = np.ma.masked_array(mesh_cube_3x3.data, mask)

        expected_mask = mesh_cube_3x3.data.mask[self.indices_to_subset,]

        subset_cube = mesh_cube_3x3[self.indices_to_subset,]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        actual_mask = subset_cube.data.mask

        np.testing.assert_array_equal(actual_mask, expected_mask)


class TestSubsetByBoolean:
    """Subset the cells 3, 4, 6, 7, 8 from the 3x3 mesh using boolean slicing.

    Expected subset:
    lat
    20                      [n]-------[n]
                             |         |
    15                       |   (4)   |
                             |         |
    10            [n]-------[n]-------[n]
                   |         |         |
    5              |   (1)   |   (3)   |
                   |         |         |
    0             [n]-------[n]-------[n]
                   |         |         |
    -5             |   (0)   |   (2)   |
                   |         |         |
    -10           [n]-------[n]-------[n]

         0    5    10   15   20   25   30  lon

    Node order is not guaranteed, but face order is.
    """

    mask_to_subset = np.zeros(9, dtype=bool)
    mask_to_subset[(3, 4, 6, 7, 8),] = True

    def test_number_of_faces(self, mesh_cube_3x3):
        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_faces = 5
        assert len(subset_mesh.face_coords.face_x.points) == expected_faces
        assert len(subset_mesh.face_coords.face_y.points) == expected_faces

    def test_face_coords(self, mesh_cube_3x3):
        """Test that the mesh faces have been subset in the order provided."""
        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_face_x = [15, 15, 25, 25, 25]
        expected_face_y = [-5, 5, -5, 5, 15]
        actual_face_x = subset_mesh.face_coords.face_x.points
        actual_face_y = subset_mesh.face_coords.face_y.points

        np.testing.assert_array_equal(actual_face_x, expected_face_x)
        np.testing.assert_array_equal(actual_face_y, expected_face_y)

    def test_number_of_nodes(self, mesh_cube_3x3):
        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_nodes = 11
        assert len(subset_mesh.node_coords.node_x.points) == expected_nodes
        assert len(subset_mesh.node_coords.node_y.points) == expected_nodes

    def test_node_coords(self, mesh_cube_3x3):
        """Test that the node locations are correct, order is not guaranteed.

        Although the node ordering is not guaranteed, the node_x and node_y arrays
        should have a consistent order, i.e. the value at index i of node_x and node_y
        both refer to the same node.
        """
        expected_nodes_x = [10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30]
        expected_nodes_y = [-10, 0, 10, -10, 0, 10, 20, -10, 0, 10, 20]
        # {(10, -10), (10, 0), (10, 10), (20, -10), (20, 0), (20, 10),
        #  (20, 20), (30, -10),(30, 0), (30, 10), (30, 20)}
        expected_nodes_xy = set(zip(expected_nodes_x, expected_nodes_y, strict=True))

        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_nodes_x = subset_mesh.node_coords.node_x.points
        actual_nodes_y = subset_mesh.node_coords.node_y.points
        actual_nodes_xy = set(zip(actual_nodes_x, actual_nodes_y, strict=True))

        assert actual_nodes_xy == expected_nodes_xy

    def test_face_node_connectivity(self, mesh_cube_3x3):
        """Test that the face node connectivity has been remapped to the new nodes.

        The actual coordinates of the nodes connected to each face should not have
        changed by subsetting the mesh. Their indices will have changed, and this
        should be reflected in a new face-node connectivity.
        """
        original_face_node_x, original_face_node_y = get_face_node_xy(
            mesh_cube_3x3.mesh
        )
        expected_face_node_x = original_face_node_x[self.mask_to_subset,]
        expected_face_node_y = original_face_node_y[self.mask_to_subset,]

        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_face_node_x, actual_face_node_y = get_face_node_xy(subset_mesh)

        np.testing.assert_array_equal(actual_face_node_x, expected_face_node_x)
        np.testing.assert_array_equal(actual_face_node_y, expected_face_node_y)

    def test_data_preserved(self, mesh_cube_3x3):
        """Test that the data array is preserved by the mesh reconstruction."""
        expected_data = mesh_cube_3x3.data[self.mask_to_subset].copy()

        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        actual_data = subset_cube.data

        np.testing.assert_array_equal(actual_data, expected_data)

    def test_mask_preserved(self, mesh_cube_3x3):
        """Test that the mask array is preserved by the mesh reconstruction."""
        mask = np.zeros_like(mesh_cube_3x3.data, dtype=bool)
        mask[(0, 3, 4),] = True
        mesh_cube_3x3.data = np.ma.masked_array(mesh_cube_3x3.data, mask)

        expected_mask = mesh_cube_3x3.data.mask[self.mask_to_subset]

        subset_cube = mesh_cube_3x3[self.mask_to_subset]
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        actual_mask = subset_cube.data.mask

        np.testing.assert_array_equal(actual_mask, expected_mask)


class TestExtractConstraint:
    """Subset the cells 1, 2, 4, 5, 7, 8 from the 3x3 mesh by constraining on latitude.

    Expected subset:
    lat
    20  [n]-------[n]-------[n]-------[n]
         |         |         |         |
    15   |   (1)   |   (3)   |   (5)   |
         |         |         |         |
    10  [n]-------[n]-------[n]-------[n]
         |         |         |         |
    5    |   (0)   |   (2)   |   (4)   |
         |         |         |         |
    0   [n]-------[n]-------[n]-------[n]

         0    5    10   15   20   25   30  lon

    Node order is not guaranteed, but face order is.
    """

    constraint = iris.Constraint(latitude=lambda y: y > -1)
    equivalent_indices = (1, 2, 4, 5, 7, 8)

    def test_number_of_faces(self, mesh_cube_3x3):
        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_faces = 6
        assert len(subset_mesh.face_coords.face_x.points) == expected_faces
        assert len(subset_mesh.face_coords.face_y.points) == expected_faces

    def test_face_coords(self, mesh_cube_3x3):
        """Test that the mesh faces have been subset in the order provided."""
        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_face_x = [5, 5, 15, 15, 25, 25]
        expected_face_y = [5, 15, 5, 15, 5, 15]
        actual_face_x = subset_mesh.face_coords.face_x.points
        actual_face_y = subset_mesh.face_coords.face_y.points

        np.testing.assert_array_equal(actual_face_x, expected_face_x)
        np.testing.assert_array_equal(actual_face_y, expected_face_y)

    def test_number_of_nodes(self, mesh_cube_3x3):
        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_nodes = 12
        assert len(subset_mesh.node_coords.node_x.points) == expected_nodes
        assert len(subset_mesh.node_coords.node_y.points) == expected_nodes

    def test_node_coords(self, mesh_cube_3x3):
        """Test that the node locations are correct, order is not guaranteed.

        Although the node ordering is not guaranteed, the node_x and node_y arrays
        should have a consistent order, i.e. the value at index i of node_x and node_y
        both refer to the same node.
        """
        expected_nodes_x = [0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30]
        expected_nodes_y = [0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20]
        # {(10, -10), (10, 0), (10, 10), (20, -10), (20, 0), (20, 10),
        #  (20, 20), (30, -10),(30, 0), (30, 10), (30, 20)}
        expected_nodes_xy = set(zip(expected_nodes_x, expected_nodes_y, strict=True))

        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_nodes_x = subset_mesh.node_coords.node_x.points
        actual_nodes_y = subset_mesh.node_coords.node_y.points
        actual_nodes_xy = set(zip(actual_nodes_x, actual_nodes_y, strict=True))

        assert actual_nodes_xy == expected_nodes_xy

    def test_face_node_connectivity(self, mesh_cube_3x3):
        """Test that the face node connectivity has been remapped to the new nodes.

        The actual coordinates of the nodes connected to each face should not have
        changed by subsetting the mesh. Their indices will have changed, and this
        should be reflected in a new face-node connectivity.
        """
        # subset face-node connectivity of the original un-split mesh
        # using indices equivalent to constraint extraction
        original_face_node_x, original_face_node_y = get_face_node_xy(
            mesh_cube_3x3.mesh
        )
        expected_face_node_x = original_face_node_x[self.equivalent_indices,]
        expected_face_node_y = original_face_node_y[self.equivalent_indices,]

        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_face_node_x, actual_face_node_y = get_face_node_xy(subset_mesh)

        np.testing.assert_array_equal(actual_face_node_x, expected_face_node_x)
        np.testing.assert_array_equal(actual_face_node_y, expected_face_node_y)

    def test_data_preserved(self, mesh_cube_3x3):
        """Test that the data array is preserved by the mesh reconstruction."""
        expected_data = mesh_cube_3x3.data[self.equivalent_indices,].copy()

        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        actual_data = subset_cube.data

        np.testing.assert_array_equal(actual_data, expected_data)

    def test_mask_preserved(self, mesh_cube_3x3):
        """Test that the mask array is preserved by the mesh reconstruction."""
        mask = np.zeros_like(mesh_cube_3x3.data, dtype=bool)
        mask[(2, 4, 8),] = True
        mesh_cube_3x3.data = np.ma.masked_array(mesh_cube_3x3.data, mask)

        expected_mask = mesh_cube_3x3.data.mask[self.equivalent_indices,]

        subset_cube = mesh_cube_3x3.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3.mesh_dim())
        actual_mask = subset_cube.data.mask

        np.testing.assert_array_equal(actual_mask, expected_mask)


class TestMultiDim:
    constraint = iris.Constraint(latitude=lambda y: y > -1)
    equivalent_indices = (1, 2, 4, 5, 7, 8)

    @pytest.fixture()
    def mesh_cube_3x3_time(self, mesh_cube_3x3):
        """Create an initial cube with a mesh to be subset, and non mesh dimension."""
        time = AuxCoord([1, 2], standard_name="time")
        mesh_cube_3x3_time1 = mesh_cube_3x3.copy()
        mesh_cube_3x3_time1.add_aux_coord(time[0])
        mesh_cube_3x3_time2 = mesh_cube_3x3.copy()
        mesh_cube_3x3_time2.add_aux_coord(time[1])

        merged_cube = CubeList([mesh_cube_3x3_time1, mesh_cube_3x3_time2]).merge()[0]
        merged_cube.data = np.arange(np.prod(merged_cube.shape)).reshape(
            merged_cube.shape
        )
        return merged_cube

    def test_number_of_faces(self, mesh_cube_3x3_time):
        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_faces = 6
        assert len(subset_mesh.face_coords.face_x.points) == expected_faces
        assert len(subset_mesh.face_coords.face_y.points) == expected_faces

    def test_face_coords(self, mesh_cube_3x3_time):
        """Test that the mesh faces have been subset in the order provided."""
        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_face_x = [5, 5, 15, 15, 25, 25]
        expected_face_y = [5, 15, 5, 15, 5, 15]
        actual_face_x = subset_mesh.face_coords.face_x.points
        actual_face_y = subset_mesh.face_coords.face_y.points

        np.testing.assert_array_equal(actual_face_x, expected_face_x)
        np.testing.assert_array_equal(actual_face_y, expected_face_y)

    def test_number_of_nodes(self, mesh_cube_3x3_time):
        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        subset_mesh = subset_cube.mesh
        expected_nodes = 12
        assert len(subset_mesh.node_coords.node_x.points) == expected_nodes
        assert len(subset_mesh.node_coords.node_y.points) == expected_nodes

    def test_node_coords(self, mesh_cube_3x3_time):
        """Test that the node locations are correct, order is not guaranteed.

        Although the node ordering is not guaranteed, the node_x and node_y arrays
        should have a consistent order, i.e. the value at index i of node_x and node_y
        both refer to the same node.
        """
        expected_nodes_x = [0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30]
        expected_nodes_y = [0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20]
        # {(10, -10), (10, 0), (10, 10), (20, -10), (20, 0), (20, 10),
        #  (20, 20), (30, -10),(30, 0), (30, 10), (30, 20)}
        expected_nodes_xy = set(zip(expected_nodes_x, expected_nodes_y, strict=True))

        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_nodes_x = subset_mesh.node_coords.node_x.points
        actual_nodes_y = subset_mesh.node_coords.node_y.points
        actual_nodes_xy = set(zip(actual_nodes_x, actual_nodes_y, strict=True))

        assert actual_nodes_xy == expected_nodes_xy

    def test_face_node_connectivity(self, mesh_cube_3x3_time):
        """Test that the face node connectivity has been remapped to the new nodes.

        The actual coordinates of the nodes connected to each face should not have
        changed by subsetting the mesh. Their indices will have changed, and this
        should be reflected in a new face-node connectivity.
        """
        # subset face-node connectivity of the original un-split mesh
        # using indices equivalent to constraint extraction
        original_face_node_x, original_face_node_y = get_face_node_xy(
            mesh_cube_3x3_time.mesh
        )
        expected_face_node_x = original_face_node_x[self.equivalent_indices,]
        expected_face_node_y = original_face_node_y[self.equivalent_indices,]

        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        subset_mesh = subset_cube.mesh
        actual_face_node_x, actual_face_node_y = get_face_node_xy(subset_mesh)

        np.testing.assert_array_equal(actual_face_node_x, expected_face_node_x)
        np.testing.assert_array_equal(actual_face_node_y, expected_face_node_y)

    def test_data_preserved(self, mesh_cube_3x3_time):
        """Test that the data array is preserved by the mesh reconstruction."""
        expected_data = mesh_cube_3x3_time.data[:, self.equivalent_indices].copy()

        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        actual_data = subset_cube.data

        np.testing.assert_array_equal(actual_data, expected_data)

    def test_mask_preserved(self, mesh_cube_3x3_time):
        """Test that the mask array is preserved by the mesh reconstruction."""
        mask = np.zeros_like(mesh_cube_3x3_time.data, dtype=bool)
        mask[(0, 1, 0), (2, 4, 8)] = True
        mesh_cube_3x3_time.data = np.ma.masked_array(mesh_cube_3x3_time.data, mask)

        expected_mask = mesh_cube_3x3_time.data.mask[:, self.equivalent_indices]

        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())
        actual_mask = subset_cube.data.mask

        np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_mesh_dimension_preserved(self, mesh_cube_3x3_time):
        """Test that reconstructing the mesh does not change the dimension order."""
        subset_cube = mesh_cube_3x3_time.extract(self.constraint)
        subset_cube = reconstruct_mesh_cube(subset_cube, mesh_cube_3x3_time.mesh_dim())

        assert subset_cube.mesh_dim() == mesh_cube_3x3_time.mesh_dim()
