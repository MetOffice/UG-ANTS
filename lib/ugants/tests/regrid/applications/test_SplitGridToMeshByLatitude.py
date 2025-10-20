# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import os
from unittest.mock import call, patch

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import Mesh

from ugants.io import load
from ugants.regrid.band_utils import mesh_to_cube
from ugants.regrid.command_line import SplitGridToMeshByLatitude
from ugants.tests import get_data_path
from ugants.tests.stock import regular_grid_global_cube, regular_lat_lon_mesh

OUTPUT_PATH = "/path/to/output.nc"
MESH_NAME = "dynamics"


@pytest.fixture()
def source_path():
    return get_data_path("non_ugrid_data.nc")


@pytest.fixture()
def sample_cubelist_from_file(source_path):
    """Load a single-element CubeList of regular lat-lon data from disk."""
    return load.cf(source_path)


@pytest.fixture()
def mesh_path():
    return get_data_path("mesh_C12.nc")


@pytest.fixture()
def mesh_C12_from_file(mesh_path):
    """Load a C12 cubedsphere mesh from disk."""
    return load.mesh(mesh_path, mesh_name=MESH_NAME)


class TestCLI:
    """Tests for application's the command line interface."""

    @pytest.fixture()
    def default_command(self, source_path, mesh_path):
        """Return default command line arguments for the fill application."""
        return [
            source_path,
            OUTPUT_PATH,
            "--target-mesh",
            mesh_path,
            "--target-mesh-name",
            MESH_NAME,
            "--number-of-bands",
            "3",
        ]

    @pytest.fixture()
    def default_app(self, default_command):
        return SplitGridToMeshByLatitude.from_command_line(default_command)

    def test_source_loaded(self, default_app, sample_cubelist_from_file):
        assert default_app.source == sample_cubelist_from_file

    def test_target_mesh_loaded(self, default_app, mesh_C12_from_file):
        assert default_app.target_mesh == mesh_C12_from_file

    def test_n_bands_assigned(self, default_app):
        assert default_app.number_of_bands == 3

    def test_no_mesh_name_given(self, source_path, mesh_path, mesh_C12_from_file):
        """Tests that the mesh will be loaded when not supplied with a mesh name."""
        command = [
            source_path,
            OUTPUT_PATH,
            "--target-mesh",
            mesh_path,
            "--number-of-bands",
            "3",
        ]
        app = SplitGridToMeshByLatitude.from_command_line(command)
        assert app.target_mesh == mesh_C12_from_file


class TestSplitGridToMeshByLatitudeSingleCube:
    """Tests for running the SplitGridToMeshByLatitude app with a single cube."""

    def test_single_cube_cast_to_cubelist(self):
        """Test that a single cube is converted to a cubelist."""
        source = regular_grid_global_cube(3, 4)
        expected = CubeList([source.copy()])
        target_mesh = regular_lat_lon_mesh()
        regrid_app = SplitGridToMeshByLatitude(source, target_mesh, 3)
        assert regrid_app.source == expected

    def test_non_global_longitude_source_cube_fails(self):
        """Test that non-global sources raise an error."""
        global_cube = regular_grid_global_cube(10, 10)
        cube_subset = global_cube.extract(
            iris.Constraint(longitude=lambda cell: cell < 180)
        )
        target_mesh = regular_lat_lon_mesh()
        with pytest.raises(
            ValueError,
            match="The provided source is not global: longitude is not circular.",
        ):
            SplitGridToMeshByLatitude(cube_subset, target_mesh, 3)

    def test_non_global_latitude_source_cube_fails(self):
        """Test that non-global sources raise an error."""
        global_cube = regular_grid_global_cube(10, 10)
        cube_subset = global_cube.extract(
            iris.Constraint(latitude=lambda cell: cell > 0)
        )
        target_mesh = regular_lat_lon_mesh()
        with pytest.raises(
            ValueError,
            match="The provided source is not global: latitude min = 18.0, "
            "latitude max = 90.0",
        ):
            SplitGridToMeshByLatitude(cube_subset, target_mesh, 3)

    def test_invalid_n_bands_fails(self):
        """Test that an error is raised when number_of_bands is not greater than 1."""
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        with pytest.raises(
            ValueError,
            match="The number of bands must be greater than 1, got 1.",
        ):
            SplitGridToMeshByLatitude(source, target_mesh, 1)

    def test_mesh_mapping(self):
        """Test that the mesh mapping cube identifies the bands correctly.

        target_mesh: 10 by 10, 1 degree per cell

        Split into 5 latitude bands, expect 2 rows per band
        """
        source = regular_grid_global_cube(10, 10)
        n_bands = 5
        target_mesh = regular_lat_lon_mesh(
            min_lon=0, max_lon=10, min_lat=0, max_lat=10, shape=(10, 10)
        )
        expected_mesh_mapping_cube = mesh_to_cube(target_mesh)

        # Set values in expected mesh mapping cube based on their cell centre latitude
        cell_centre_latitudes = expected_mesh_mapping_cube.coord("latitude").points
        expected_mesh_mapping_cube.data[
            np.logical_and(0 < cell_centre_latitudes, cell_centre_latitudes < 2)
        ] = 0
        expected_mesh_mapping_cube.data[
            np.logical_and(2 < cell_centre_latitudes, cell_centre_latitudes < 4)
        ] = 1
        expected_mesh_mapping_cube.data[
            np.logical_and(4 < cell_centre_latitudes, cell_centre_latitudes < 6)
        ] = 2
        expected_mesh_mapping_cube.data[
            np.logical_and(6 < cell_centre_latitudes, cell_centre_latitudes < 8)
        ] = 3
        expected_mesh_mapping_cube.data[
            np.logical_and(8 < cell_centre_latitudes, cell_centre_latitudes < 10)
        ] = 4

        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        actual_mesh_mapping_cube = app.mesh_mapping_cube

        assert actual_mesh_mapping_cube == expected_mesh_mapping_cube

    def test_mesh_bands(self):
        """Test that the correct mesh bands are generated.

        Latitude bounds will be:
        [(-90, -30), (-30, +30), (+30, +90)]

        Expected bands are of shape (18, 6)
        Node spacing 10 degrees
        Face centres are offset by 5 degrees from nodes

        The following diagram shows a segment of the mesh:
           |           |            |
        (0,-70)-----(10,-70)-----(20,-70)-----
           |           |            |
           |  (5,-75)  |  (10,-75)  |  (15,-75)
           |           |            |
        (0,-80)-----(10,-80)-----(20,-80)-----
           |           |            |
           |  (5,-85)  |  (10,-85)  |  (15,-85)
           |           |            |
        (0,-90)-----(10,-90)-----(20,-90)-----
        """
        source = regular_grid_global_cube(10, 10)
        n_bands = 3
        target_mesh = regular_lat_lon_mesh(min_lat=-90.0, max_lat=+90.0, shape=(18, 18))

        expected_mesh_extents = [
            {
                "max_node_lat": -30.0,
                "min_node_lat": -90.0,
                "max_node_lon": 180.0,
                "min_node_lon": 0.0,
                "max_face_lat": -35.0,
                "min_face_lat": -85.0,
                "max_face_lon": 175.0,
                "min_face_lon": 5.0,
            },
            {
                "max_node_lat": 30.0,
                "min_node_lat": -30.0,
                "max_node_lon": 180.0,
                "min_node_lon": 0.0,
                "max_face_lat": 25.0,
                "min_face_lat": -25.0,
                "max_face_lon": 175.0,
                "min_face_lon": 5.0,
            },
            {
                "max_node_lat": 90.0,
                "min_node_lat": 30.0,
                "max_node_lon": 180.0,
                "min_node_lon": 0.0,
                "max_face_lat": 85.0,
                "min_face_lat": 35.0,
                "max_face_lon": 175.0,
                "min_face_lon": 5.0,
            },
        ]
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        actual_mesh_bands = app.mesh_bands
        assert len(actual_mesh_bands) == n_bands

        actual_mesh_extents = [
            _get_mesh_extents(mesh_band) for mesh_band in actual_mesh_bands
        ]
        assert actual_mesh_extents == expected_mesh_extents

    def test_source_bands_latitudes(self):
        """Test that the source bands are created with the correct latitudes.

        Sample data has dimensions:
        latitude = 144
        longitude = 192

        Latitude bounds of the mesh will be:
        [(-90, -60), (-60, -30), (-30, 0), (0, +30), (+30, +60), (+60, +90)]

        The source bands must cover these bounds:

        90 ---------------------------=== 90
                                       |
                                       |  === >60
        60 --------------------------- |   |
                                  <60 ===  |
                                  >30 ===  |
        30 --------------------------- |   |
                                       |  === <30
                                       |
         0 --------------------------- |
                                  < 0 ===

        --- represents the mesh band boundary
        ===
         |  represents the extent of a source band
        ===
        """
        source = regular_grid_global_cube(144, 192)
        n_bands = 6
        target_mesh = regular_lat_lon_mesh(min_lat=-90.0, max_lat=+90.0, shape=(18, 18))
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        # Note app.results is a list of CubeLists. The length of the list is n_bands (6)
        # and the length of each CubeList is 1 (since the sample_cubelist has length 1).
        actual_source_bands = app.results
        assert len(actual_source_bands) == n_bands
        mesh_bounds = [
            (-90, -60),
            (-60, -30),
            (-30, 0),
            (0, +30),
            (+30, +60),
            (+60, +90),
        ]
        actual_grid_extents = [
            _get_grid_extents(source_band[0]) for source_band in actual_source_bands
        ]
        for mesh_bound, grid_extent in zip(
            mesh_bounds, actual_grid_extents, strict=True
        ):
            assert grid_extent["min_lat"] <= mesh_bound[0]
            assert grid_extent["max_lat"] >= mesh_bound[1]

    def test_source_bands_longitudes(self):
        """Test that the source bands are created with the correct longitudes.

        Sample data has dimensions:
        latitude = 144
        longitude = 192

        Longitude coordinate domain: -180 to +180
        """
        source = regular_grid_global_cube(144, 192)
        n_bands = 6
        target_mesh = regular_lat_lon_mesh(min_lat=-90.0, max_lat=+90.0, shape=(18, 18))
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        # Note app.results is a list of CubeLists. The length of the list is n_bands (6)
        # and the length of each CubeList is 1 (since the sample_cubelist has length 1).
        actual_source_bands = app.results
        assert len(actual_source_bands) == n_bands

        actual_grid_extents = [
            _get_grid_extents(source_band[0]) for source_band in actual_source_bands
        ]
        for grid_extent in actual_grid_extents:
            assert grid_extent["min_lon"] == -180.0
            assert grid_extent["max_lon"] == 180.0

    def test_source_bands_number_attribute(self):
        """Test that the band_number attribute has been added to each source band."""
        source = regular_grid_global_cube(10, 10)
        n_bands = 6
        target_mesh = regular_lat_lon_mesh(min_lat=-90.0, max_lat=+90.0, shape=(18, 18))
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        # Note app.results is a list of CubeLists. The length of the list is n_bands (6)
        # and the length of each CubeList is 1 (since the sample_cubelist has length 1).
        for band_num, source_band in enumerate(app.results):
            assert source_band[0].attributes["band_number"] == band_num


def _get_mesh_extents(mesh: Mesh):
    max_node_lat = max(mesh.node_coords.node_y.points)
    min_node_lat = min(mesh.node_coords.node_y.points)
    max_node_lon = max(mesh.node_coords.node_x.points)
    min_node_lon = min(mesh.node_coords.node_x.points)

    max_face_lat = max(mesh.face_coords.face_y.points)
    min_face_lat = min(mesh.face_coords.face_y.points)
    max_face_lon = max(mesh.face_coords.face_x.points)
    min_face_lon = min(mesh.face_coords.face_x.points)

    return {
        "max_node_lat": max_node_lat,
        "min_node_lat": min_node_lat,
        "max_node_lon": max_node_lon,
        "min_node_lon": min_node_lon,
        "max_face_lat": max_face_lat,
        "min_face_lat": min_face_lat,
        "max_face_lon": max_face_lon,
        "min_face_lon": min_face_lon,
    }


def _get_grid_extents(cube: Cube):
    latitude = cube.coord(axis="y").copy()
    if not latitude.has_bounds():
        latitude.guess_bounds()
    max_lat = max(latitude.bounds.flatten())
    min_lat = min(latitude.bounds.flatten())

    longitude = cube.coord(axis="x").copy()
    if not longitude.has_bounds():
        longitude.guess_bounds()
    max_lon = max(longitude.bounds.flatten())
    min_lon = min(longitude.bounds.flatten())
    return {
        "max_lat": max_lat,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "min_lon": min_lon,
    }


class TestSplitGridToMeshByLatitudeMultiCube:
    """Tests for the SplitGridToMeshByLatitude app with multiple source cubes."""

    n_bands = 6

    @pytest.fixture()
    def multi_source_cubelist(self):
        first_cube = regular_grid_global_cube(20, 20)
        second_cube = first_cube.copy() + 1
        second_cube.rename("additional_sample_data")
        source_cubes = CubeList([first_cube, second_cube])
        return source_cubes

    def test_different_source_grids_raises_error(self):
        """Test that an error is raised when cubes have different horizontal grids."""
        first_cube = regular_grid_global_cube(10, 10)
        second_cube = regular_grid_global_cube(11, 11)
        source = CubeList([first_cube, second_cube])
        target_mesh = regular_lat_lon_mesh()
        with pytest.raises(
            ValueError, match="Not all source cubes have the same horizontal grid."
        ):
            SplitGridToMeshByLatitude(source, target_mesh, 3)

    def test_number_of_outputs(self, multi_source_cubelist):
        """Test that the expected number of outputs has been produced.

        Expect n_bands CubeLists, each containing 2 cubes.
        """
        target_mesh = regular_lat_lon_mesh()
        app = SplitGridToMeshByLatitude(
            multi_source_cubelist, target_mesh, self.n_bands
        )
        app.run()
        assert len(app.results) == self.n_bands
        for cubelist_band in app.results:
            assert len(cubelist_band) == 2

    def test_multi_source_split_is_consistent_with_single_source_split(
        self, multi_source_cubelist
    ):
        """Test that splitting multiple source cubes gives consistent results to splitting single cubes individually."""  # noqa: E501
        # Extract the two cubes from the multi cube source and split them separately
        target_mesh = regular_lat_lon_mesh()

        sample_data_source = multi_source_cubelist.extract_cube("sample_data")
        sample_data_split_app = SplitGridToMeshByLatitude(
            sample_data_source, target_mesh, self.n_bands
        )
        sample_data_split_app.run()
        expected_sample_data_bands = sample_data_split_app.results

        additional_sample_data_source = multi_source_cubelist.extract_cube(
            "additional_sample_data"
        )
        additional_sample_data_split_app = SplitGridToMeshByLatitude(
            additional_sample_data_source, target_mesh, self.n_bands
        )
        additional_sample_data_split_app.run()
        additional_sample_data_bands = additional_sample_data_split_app.results

        # Split the multi cube source in one go
        multi_cube_split = SplitGridToMeshByLatitude(
            multi_source_cubelist, target_mesh, self.n_bands
        )
        multi_cube_split.run()

        # Extract the multi cube split results into two separate lists, one per field
        actual_sample_data_bands = [
            cubelist.extract("sample_data") for cubelist in multi_cube_split.results
        ]
        actual_additional_sample_data_bands = [
            cubelist.extract("additional_sample_data")
            for cubelist in multi_cube_split.results
        ]

        # Assert that all CubeLists are consistent between the two approaches
        for actual_band, expected_band in zip(
            actual_sample_data_bands, expected_sample_data_bands, strict=True
        ):
            assert actual_band == expected_band

        for actual_band, expected_band in zip(
            actual_additional_sample_data_bands,
            additional_sample_data_bands,
            strict=True,
        ):
            assert actual_band == expected_band


class TestSaveCalls:
    """Tests for the calls to relevant savers in the app's save method."""

    ugrid_save_target = "ugants.regrid.command_line.save.ugrid"
    mesh_save_target = "ugants.regrid.command_line.save_mesh"
    netcdf_save_target = "ugants.regrid.command_line.save_netcdf"

    @patch(mesh_save_target)
    @patch(netcdf_save_target)
    def test_save_mesh_mapping_cube(
        self,
        mock_netcdf_save,
        mock_mesh_save,
    ):
        """Test that the UGrid saver is called with the mesh mapping cube."""
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        n_bands = 5
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        app.output = OUTPUT_PATH

        expected_call = app.mesh_mapping_cube, os.path.join(
            OUTPUT_PATH, "mesh_band_mapping.nc"
        )

        with patch(self.ugrid_save_target) as mock_ugrid_save:
            app.save()
        mock_ugrid_save.assert_called_once_with(*expected_call)

    @patch(ugrid_save_target)
    @patch(netcdf_save_target)
    def test_save_mesh_band(self, mock_netcdf_save, mock_ugrid_save):
        """Test that the mesh saver is called with the mesh bands."""
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        n_bands = 5
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        app.output = OUTPUT_PATH

        output_paths = [
            os.path.join(OUTPUT_PATH, f"mesh_band_{band_number}.nc")
            for band_number in range(n_bands)
        ]
        expected_calls = [
            call(mesh_band, output_path)
            for mesh_band, output_path in zip(app.mesh_bands, output_paths, strict=True)
        ]

        with patch(self.mesh_save_target) as mock_mesh_save:
            app.save()

        mock_mesh_save.assert_has_calls(expected_calls)

    @patch(ugrid_save_target)
    @patch(mesh_save_target)
    def test_save_source_band(self, mock_mesh_save, mock_ugrid_save):
        """Test that the netcdf saver is called with the source bands."""
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        n_bands = 5
        app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
        app.run()
        app.output = OUTPUT_PATH

        output_paths = [
            os.path.join(OUTPUT_PATH, f"source_band_{band_number}.nc")
            for band_number in range(n_bands)
        ]
        expected_calls = [
            call(source_band, output_path)
            for source_band, output_path in zip(app.results, output_paths, strict=True)
        ]

        with patch(self.netcdf_save_target) as mock_netcdf_save:
            app.save()

        mock_netcdf_save.assert_has_calls(expected_calls)

    def test_save_without_output_fails(self):
        """Test that an error is raised when attempting to save before output is set."""
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        app = SplitGridToMeshByLatitude(source, target_mesh, 3)
        assert app.output is None
        with pytest.raises(
            ValueError, match="No output directory location has been set."
        ):
            app.save()

    def test_save_without_mesh_mapping_fails(self):
        """Test that an error is raised when attempting to save before mesh_mapping_cube has been set."""  # noqa: E501
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        app = SplitGridToMeshByLatitude(source, target_mesh, 3)
        app.output = "dummy_output"
        app.mesh_bands = "dummy_mesh_bands"
        app.results = "dummy_source_bands"
        assert not hasattr(app, "mesh_mapping_cube")

        with pytest.raises(
            ValueError,
            match="The application has not yet been run, mesh_mapping_cube is not set.",
        ):
            app.save()

    def test_save_without_mesh_bands_fails(self):
        """Test that an error is raised when attempting to save before mesh_bands has been set."""  # noqa: E501
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        app = SplitGridToMeshByLatitude(source, target_mesh, 3)
        app.output = "dummy_output"
        app.mesh_mapping_cube = "dummy_mesh_mapping_cube"
        app.results = "dummy_source_bands"
        assert not hasattr(app, "mesh_bands")

        with pytest.raises(
            ValueError,
            match="The application has not yet been run, mesh_bands is not set.",
        ):
            app.save()

    def test_save_without_source_bands_fails(self):
        """Test that an error is raised when attempting to save before source_bands has been set."""  # noqa: E501
        source = regular_grid_global_cube(3, 4)
        target_mesh = regular_lat_lon_mesh()
        app = SplitGridToMeshByLatitude(source, target_mesh, 3)
        app.output = "dummy_output"
        app.mesh_bands = "dummy_mesh_bands"
        app.mesh_mapping_cube = "dummy_mesh_mapping_cube"
        assert app.results is None

        with pytest.raises(
            ValueError,
            match="The application has not yet been run, results is not set.",
        ):
            app.save()
