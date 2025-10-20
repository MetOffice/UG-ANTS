# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import CubeList
from iris.experimental.ugrid.mesh import Connectivity, Mesh
from numpy import ndarray
from numpy.ma import MaskedArray, is_masked, masked, masked_array, masked_less

from ugants.filter.example_filters import FaceNeighbourhoodFilter
from ugants.tests.stock import sample_mesh, sample_mesh_cube


def mesh_with_edges_but_no_faces():
    # Code mostly pinched from iris.tests.stock.mesh.sample_mesh
    # -- which, unfortunately, can currently *not* produce a mesh with no faces.
    n_nodes = 5
    n_edges = 3
    node_x = AuxCoord(
        1100 + np.arange(n_nodes), standard_name="longitude", units="degrees_east"
    )
    node_y = AuxCoord(
        1200 + np.arange(n_nodes), standard_name="latitude", units="degrees_north"
    )

    # Define a rather arbitrary edge-nodes connectivity.
    edge_nodes = Connectivity(
        [[0, 1], [1, 2], [1, 3]], cf_role="edge_node_connectivity"
    )

    edge_x = AuxCoord(2100 + np.arange(n_edges), standard_name="longitude")
    edge_y = AuxCoord(2200 + np.arange(n_edges), standard_name="latitude")

    mesh = Mesh(
        topology_dimension=1,
        node_coords_and_axes=[(node_x, "x"), (node_y, "y")],
        connectivities=[edge_nodes],
        edge_coords_and_axes=[(edge_x, "x"), (edge_y, "y")],
    )
    return mesh


class TestFaceNeighbourhoodFilter:
    def test_nofaces__fail(self):
        # Create a test cube, based on a mesh with edges but no faces.
        mesh = mesh_with_edges_but_no_faces()
        cube = sample_mesh_cube(mesh=mesh, location="edge")

        with pytest.raises(Exception) as error:
            FaceNeighbourhoodFilter(cube)

        expect = (
            "ValueError('A FaceNeighbourhoodFilter can only work with face-located "
            'data. The provided reference cube, "mesh_phenom", has location '
            r"= \'edge\'.')"
        )
        assert repr(error.value) == expect

    def test_no_face_face_connectivity__fail(self):
        # NOTE: as it happens, the default sample mesh has no face-face connectivity.
        mesh = sample_mesh()
        cube = sample_mesh_cube(mesh=mesh)

        with pytest.raises(Exception) as error:
            FaceNeighbourhoodFilter(cube)

        expect = (
            "ValueError('A FaceNeighbourhoodFilter requires face connectivity data. "
            'The provided reference cube, "mesh_phenom", has a mesh with an empty '
            '"mesh.face_face_connectivity".\')'
        )
        assert repr(error.value) == expect

    def test_no_face_neighbours(self):
        """Test filtering on a mesh with no (valid) face neighbours."""
        number_of_faces = 5
        mesh = sample_mesh(n_faces=number_of_faces)

        # Make an all-empty face-face connectivity array.
        # Allow 3 neighbours per face, but make them all 'missing'.
        connectivity_shape = (number_of_faces, 3)
        empty_connectivity_array = masked_array(
            np.zeros(connectivity_shape, dtype=int),
            mask=np.ones(connectivity_shape, dtype=bool),
        )
        face_face_connectivity = Connectivity(
            empty_connectivity_array, cf_role="face_face_connectivity"
        )
        mesh.add_connectivities(face_face_connectivity)

        # Create a cube on that mesh, and install some non-zero data.
        cube = sample_mesh_cube(mesh=mesh)
        cube.data = np.arange(cube.data.size).reshape(cube.shape)

        # Form a filter which scales the central values.  It should get *no* neighbour
        # contributions, since there are no neighbours ...
        central_fraction = 0.123
        filter_instance = FaceNeighbourhoodFilter(
            cube, central_fraction=central_fraction, neighbours_fraction=1000.0
        )

        result = filter_instance(cube)

        # Check that the result is just the input * central-factor
        assert result == cube.copy(data=central_fraction * cube.data)

    def _neighbourhood_calculation_testcase(self):
        # Create a specific, small, connectivity testcase for calculation testing.
        # ***NOTE***
        # It really doesn't matter that this connectivity makes no topological "sense",
        # e.g. that the neighbour-relationship constructed below is not symmetrical.
        # The operation does not rely on any such consistency or "meaning".

        n_faces = 4
        # N.B. use -1 for "missing" : the number of neighbours varies.
        face_face_indices = masked_less(
            [
                [1, 2, 3],
                [0, 2, -1],
                [-1, -1, -1],
                [-1, 2, -1],
            ],
            0,
        )

        # Define specific input values, and the filtering fractions to be used.
        face_values = np.array([1.0, 2, 3, 4])
        central_fraction = 10.0
        neighbours_fraction = 0.1
        # Define the expected result of the calculation, based on the above numbers.
        expected_results = np.array(
            [
                10.0 + 0.1 * (2 + 3 + 4) / 3,
                20.0 + 0.1 * (1 + 3) / 2,
                30.0,
                40.0 + 0.1 * 3,
            ]
        )

        # Create a mesh with this connectivity
        mesh = sample_mesh(n_faces=n_faces, n_edges=0)
        mesh.add_connectivities(
            Connectivity(face_face_indices, cf_role="face_face_connectivity")
        )
        # Create a cube on the mesh with specific data, and a filter based on that.
        cube = sample_mesh_cube(mesh=mesh, n_z=1)
        cube = cube[0]  # Remove the 'Z' dimension which sample_mesh_cube produces.
        cube.data = face_values

        # Create the test filter
        filter_instance = FaceNeighbourhoodFilter(
            cube,
            central_fraction=central_fraction,
            neighbours_fraction=neighbours_fraction,
        )

        return (
            cube,
            central_fraction,
            neighbours_fraction,
            filter_instance,
            expected_results,
        )

    def test_neighbourhood_calculation(self):
        (
            cube,
            central_fraction,
            neighbours_fraction,
            filter_instance,
            expected_results,
        ) = self._neighbourhood_calculation_testcase()

        result = filter_instance(cube)

        assert result == cube.copy(data=expected_results)

    def test_calculation_with_extra_dims(self):
        """Check that the test filter correctly handles extra cube dimensions."""
        (
            cube,
            central_fraction,
            neighbours_fraction,
            filter_instance,
            expected_results,
        ) = self._neighbourhood_calculation_testcase()

        # Create a cube with expanded dims by merging on 2 scalar coords.
        cube.add_aux_coord(AuxCoord([0.0], long_name="extra_dim_1"))
        cube.add_aux_coord(AuxCoord([0.0], long_name="extra_dim_2"))
        length_extra_dim_1, length_extra_dim_2 = (3, 4)
        multiple_cubes = [
            cube.copy()
            for __1 in range(length_extra_dim_1)
            for __2 in range(length_extra_dim_2)
        ]
        for i_x in range(length_extra_dim_1):
            for i_y in range(length_extra_dim_2):
                cube = multiple_cubes[i_y * length_extra_dim_1 + i_x]
                cube.coord("extra_dim_1").points = [i_x]
                cube.coord("extra_dim_2").points = [i_y]
        # merge to create a single combined cube with 2 additional dimensions.
        multidimensional_cube = CubeList(multiple_cubes).merge_cube()
        # This produces a cube [extra_1, extra_2, mesh]
        # transpose this to put one dimension before, and one after, the mesh dimension
        # i.e. [extra_1, mesh, extra_2]
        multidimensional_cube.transpose([0, 2, 1])

        # Zero the data of the input cube except for the the [0, :, 0],  'slice', so
        # that we can check correct replication of the calculation across dimensions.
        multidimensional_cube.data[1:] = 0.0
        multidimensional_cube.data[:, :, 1:] = 0.0

        # Get the calculation result (just the data array)
        result_data = filter_instance(multidimensional_cube).data

        # work out what the result ought to be, i.e. one slice equal to the 'basic'
        # result and the rest zeros (like the input data).
        mesh_length = cube.shape[0]
        expected_data = np.zeros((length_extra_dim_1, mesh_length, length_extra_dim_2))
        expected_data[0, :, 0] = expected_results

        # confirm correct replication of results.
        assert np.all(result_data == expected_data)

    def test_masked_input_datapoints(self):
        """
        Check what the calculation does with masked data points.

        The actual behaviour is effectively to **treat masked points as zeros**.

        N.B. this is not what you might expect :  It is because the implementation
        computes the number of neighbours in advance, simply to demonstrate the use of
        the "precalculate" step.
        Whereas, a more practical implementation would probably compute this for
        *each* input, and exclude masked neighbours from the average, which gives
        different results.
        """
        (
            cube,
            central_fraction,
            neighbours_fraction,
            filter_instance,
            expected_results,
        ) = self._neighbourhood_calculation_testcase()

        # Set up an alternative testcube with just face #2 masked out.
        # N.B. from the above testcase, this influences several output values.
        cube_masked = cube.copy()
        cube_masked.data = masked_array(cube.data)
        cube_masked.data[2] = masked
        # Construct a similar input with the face#2 value set to 0.
        cube_zeroed = cube.copy()
        cube_zeroed.data[2] = 0.0

        # The thing to test here is that, in the case of this particular (arguably odd)
        # calculation, masked input points behave just like zeros.
        result_masked = filter_instance(cube_masked)
        result_zeroed = filter_instance(cube_zeroed)

        assert result_masked == result_zeroed

    def test_unmasked_input_result_type(self):
        """
        Check the output type with un-masked input.

        That is : masked data with no masked points.
        """
        (
            cube,
            central_fraction,
            neighbours_fraction,
            filter_instance,
            expected_results,
        ) = self._neighbourhood_calculation_testcase()

        result_array = filter_instance(cube).data
        result = (type(cube.data), type(result_array), is_masked(result_array))

        expected = (ndarray, MaskedArray, False)
        assert result == expected

    def test_masked_input_result_type(self):
        """Confirm that masked input produces a masked output."""
        (
            cube,
            central_fraction,
            neighbours_fraction,
            filter_instance,
            expected_results,
        ) = self._neighbourhood_calculation_testcase()

        cube.data = masked_array(cube.data, [False, False, True, True])

        result_array = filter_instance(cube).data

        assert type(result_array) == MaskedArray
