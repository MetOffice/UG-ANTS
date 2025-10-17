# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import os
import re
from tempfile import TemporaryDirectory

import iris.cube
import numpy as np
import pytest

from ugants.io import load, save
from ugants.regrid.command_line import RecombineMeshBands
from ugants.tests.stock import mesh_cube


@pytest.fixture()
def mesh_mapping_cube():
    """Return a mesh mapping cube mapping 9 faces to 3 unequally sized bands.

    The first 4 faces map to band 1, the next 3 faces map to band 0, and the final 2
    faces map to band 2.

    mesh_mapping_data = [1, 1, 1, 1, 0, 0, 0, 2, 2]
    """
    mesh_mapping_cube = mesh_cube(number_of_levels=0, n_faces=9)
    mesh_mapping_cube.data[:4] = 1
    mesh_mapping_cube.data[4:7] = 0
    mesh_mapping_cube.data[7:9] = 2
    return mesh_mapping_cube


@pytest.fixture()
def single_dim_bands():
    """Return a cubelist containing three band cubes.

    Each band cube has an unstructured mesh (dimension 0) with data on the faces.

    Band 0 has an unstructured dimension of length 3.
    Band 1 has an unstructured dimension of length 4.
    Band 2 has an unstructured dimension of length 2.
    """
    band_0 = mesh_cube(number_of_levels=0, n_faces=3)
    band_0.data = np.arange(3)
    band_0.attributes["band_number"] = 0
    band_0.rename("single_dim_data")

    band_1 = mesh_cube(number_of_levels=0, n_faces=4)
    band_1.data = np.arange(3, 7)
    band_1.attributes["band_number"] = 1
    band_1.rename("single_dim_data")

    band_2 = mesh_cube(number_of_levels=0, n_faces=2)
    band_2.data = np.arange(7, 9)
    band_2.attributes["band_number"] = 2
    band_2.rename("single_dim_data")

    return iris.cube.CubeList([band_0, band_1, band_2])


@pytest.fixture()
def multi_dim_bands():
    """Return a cubelist containing three band cubes.

    Each band cube has a structured "level" coordinate of length 2 (dimension 0)
    and an unstructured mesh (dimension 1) with data on the faces.

    Band 0 has an unstructured dimension of length 3.
    Band 1 has an unstructured dimension of length 4.
    Band 2 has an unstructured dimension of length 2.
    """
    band_0 = mesh_cube(number_of_levels=2, n_faces=3)
    band_0.data = np.arange(6).reshape(2, 3)
    band_0.attributes["band_number"] = 0
    band_0.rename("multi_dim_data")

    band_1 = mesh_cube(number_of_levels=2, n_faces=4)
    band_1.data = np.arange(6, 14).reshape(2, 4)
    band_1.attributes["band_number"] = 1
    band_1.rename("multi_dim_data")

    band_2 = mesh_cube(number_of_levels=2, n_faces=2)
    band_2.data = np.arange(14, 18).reshape(2, 2)
    band_2.attributes["band_number"] = 2
    band_2.rename("multi_dim_data")

    return iris.cube.CubeList([band_0, band_1, band_2])


class TestCLI:
    """Tests for the application's command line interface."""

    @pytest.fixture()
    def bands_dir(self, single_dim_bands, mesh_mapping_cube):
        """Save some synthetic data to files in a temporary directory."""
        with TemporaryDirectory() as temp_dir:
            save.ugrid(
                single_dim_bands[0], os.path.join(temp_dir, "regridded_band_0.nc")
            )
            save.ugrid(
                single_dim_bands[1], os.path.join(temp_dir, "regridded_band_1.nc")
            )
            save.ugrid(
                single_dim_bands[2], os.path.join(temp_dir, "regridded_band_2.nc")
            )
            save.ugrid(mesh_mapping_cube, os.path.join(temp_dir, "mesh_mapping.nc"))
            yield temp_dir

    def test_mesh_mapping_loaded(self, bands_dir):
        """Test that the mesh mapping is correctly loaded."""
        bands_path = os.path.join(bands_dir, "regridded_band_*.nc")
        mesh_mapping_path = os.path.join(bands_dir, "mesh_mapping.nc")
        expected_mesh_mapping = load.ugrid(mesh_mapping_path)

        command_line_args = [mesh_mapping_path, bands_path, "output/path"]
        app = RecombineMeshBands.from_command_line(command_line_args)

        assert app.mesh_mapping == expected_mesh_mapping

    def test_bands_loaded(self, bands_dir):
        """Test that multiple bands are loaded from the same directory."""
        bands_path = os.path.join(bands_dir, "regridded_band_*.nc")
        mesh_mapping_path = os.path.join(bands_dir, "mesh_mapping.nc")
        expected_bands = load.ugrid(bands_path)

        command_line_args = [
            mesh_mapping_path,
            bands_path,
            "output/path",
        ]
        app = RecombineMeshBands.from_command_line(command_line_args)

        assert app.bands == expected_bands


class TestValidation:
    def test_mesh_mapping_has_no_mesh(self, mesh_mapping_cube, single_dim_bands):
        """Test that providing a mesh mapping cube without a mesh raises an error."""
        # Subset mesh mapping cube to break the mesh
        mesh_mapping_cube = mesh_mapping_cube[:-1]
        assert mesh_mapping_cube.mesh is None
        with pytest.raises(
            ValueError, match="The provided mesh_mapping does not contain a mesh."
        ):
            RecombineMeshBands(mesh_mapping_cube, single_dim_bands)

    def test_mesh_mapping_has_extra_dimension(self, single_dim_bands):
        """Test that providing a mesh mapping cube with more than one dimension raises an error."""  # noqa: E501
        mesh_mapping = mesh_cube(number_of_levels=2)
        with pytest.raises(
            ValueError,
            match="The provided mesh_mapping should have 1 dimension, got 2.",
        ):
            RecombineMeshBands(mesh_mapping, single_dim_bands)

    def test_invalid_band_numbers_single_variable(
        self, mesh_mapping_cube, single_dim_bands
    ):
        """Test that providing an inconsistent set of band numbers raises an error.

        In this case, the application expects bands 0, 1, and 2, but only bands 0 and 2
        are provided.
        """
        # Remove band 1 cube from bands cubelist
        single_dim_bands.pop(1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Inconsistent mesh bands provided for single_dim_data: expected "
                "[0, 1, 2], got [0, 2]"
            ),
        ):
            RecombineMeshBands(mesh_mapping_cube, single_dim_bands)

    def test_invalid_band_numbers_multi_variable(
        self, mesh_mapping_cube, single_dim_bands, multi_dim_bands
    ):
        """Test that providing an inconsistent set of band numbers raises an error.

        In this case, one variable contains bands 0, 1, 2 as expected,
        but the other variable contains bands 0 and 1 only.
        """
        # Remove band 2 cube from the multi dim bands cubelist
        multi_dim_bands.pop(2)
        all_bands = multi_dim_bands + single_dim_bands
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Inconsistent mesh bands provided for multi_dim_data: expected "
                "[0, 1, 2], got [0, 1]"
            ),
        ):
            RecombineMeshBands(mesh_mapping_cube, all_bands)

    def test_inconsistent_mesh_dimension_length_single_variable(
        self, mesh_mapping_cube, single_dim_bands
    ):
        """Test that an error is raised when there is inconsistency in the mesh dimension lengths.

        The mesh mapping has mesh dimension length of 9, and the provided bands have a
        total mesh dimension length of 10.
        """  # noqa: E501
        # Replace the original band 2 cube of length 2 with a cube of length 3
        replacement_band_2 = mesh_cube(number_of_levels=0, n_faces=3)
        replacement_band_2.attributes["band_number"] = 2
        replacement_band_2.rename("single_dim_data")
        single_dim_bands[2] = replacement_band_2

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Inconsistent unstructured dimension lengths for single_dim_data "
                "bands. Provided bands have unstructured dimensions of lengths "
                "[3, 4, 3] giving a total of 10, whereas mesh mapping has length 9.",
            ),
        ):
            RecombineMeshBands(mesh_mapping_cube, single_dim_bands)

    def test_inconsistent_mesh_dimension_length_multi_variable(
        self, mesh_mapping_cube, single_dim_bands, multi_dim_bands
    ):
        """Test that an error is raised when there is inconsistency in the mesh dimension lengths.

        The mesh mapping has mesh dimension length of 9, and the provided bands for one
        of the variables have a total mesh dimension length of 10.
        """  # noqa: E501
        # Replace the original band 1 cube of length 4 with a cube of length 3
        replacement_band_1 = mesh_cube(number_of_levels=3, n_faces=3)
        replacement_band_1.attributes["band_number"] = 1
        replacement_band_1.rename("multi_dim_data")
        multi_dim_bands[1] = replacement_band_1

        all_bands = single_dim_bands + multi_dim_bands

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Inconsistent unstructured dimension lengths for multi_dim_data "
                "bands. Provided bands have unstructured dimensions of lengths "
                "[3, 3, 2] giving a total of 8, whereas mesh mapping has length 9.",
            ),
        ):
            RecombineMeshBands(mesh_mapping_cube, all_bands)


class TestRecombineSingleDimCube:
    """Tests for recombining into a single cube with an unstructured dimension only."""

    def test_recombined_data(self, mesh_mapping_cube, single_dim_bands):
        """Test that the data has been recombined into the correct cells."""
        expected_data = np.empty(9)

        # Populate band 0 data at indices 4, 5, 6
        expected_data[4:7] = np.arange(3)
        # Populate band 1 data at indices 0, 1, 2, 3
        expected_data[:4] = np.arange(3, 7)
        # Populate band 2 data at indices 7, 8
        expected_data[7:] = np.arange(7, 9)

        app = RecombineMeshBands(mesh_mapping_cube, single_dim_bands)
        app.run()
        assert len(app.results) == 1
        result = app.results.extract_cube("single_dim_data")
        np.testing.assert_array_equal(result.data, expected_data)

    def test_recombined_metadata(self, mesh_mapping_cube, single_dim_bands):
        """Test that the recombined result inherits metadata from the bands.

        The band number should be discarded.
        """
        expected_metadata = single_dim_bands[0].copy().metadata
        expected_metadata.attributes.pop("band_number")

        app = RecombineMeshBands(mesh_mapping_cube, single_dim_bands)
        app.run()
        result = app.results.extract_cube("single_dim_data")
        assert result.metadata == expected_metadata


class TestRecombineMultiDimCube:
    """Tests for recombining a cube with an unstructured and a structured dimension."""

    def test_recombine_data(self, mesh_mapping_cube, multi_dim_bands):
        """Test that the data has been recombined into the correct cells."""
        expected_data = np.empty((2, 9))

        # Populate band 0 data at indices 4, 5, 6
        expected_data[:, 4:7] = np.arange(6).reshape(2, 3)
        # Populate band 1 data at indices 0, 1, 2, 3
        expected_data[:, :4] = np.arange(6, 14).reshape(2, 4)
        # Populate band 2 data at indices 7, 8
        expected_data[:, 7:9] = np.arange(14, 18).reshape(2, 2)

        app = RecombineMeshBands(mesh_mapping_cube, multi_dim_bands)
        app.run()
        assert len(app.results) == 1
        result = app.results.extract_cube("multi_dim_data")
        np.testing.assert_array_equal(result.data, expected_data)

    def test_recombine_metadata(self, mesh_mapping_cube, multi_dim_bands):
        """Test that the recombined result inherits metadata from the bands.

        The band number should be discarded.
        """
        expected_metadata = multi_dim_bands[0].copy().metadata
        expected_metadata.attributes.pop("band_number")

        app = RecombineMeshBands(mesh_mapping_cube, multi_dim_bands)
        app.run()
        result = app.results.extract_cube("multi_dim_data")
        assert result.metadata == expected_metadata


class TestRecombineMultiDimCubes:
    """Tests for recombining multiple cubes.

    Both cubes have the same unstructured dimension, but one of them also has a
    structured dimension.
    """

    def test_recombine_data(self, mesh_mapping_cube, multi_dim_bands, single_dim_bands):
        """Test that the data has been recombined into the correct cells."""
        expected_data_multi_dim = np.empty((2, 9))

        # Populate band 0 data at indices 4, 5, 6
        expected_data_multi_dim[:, 4:7] = np.arange(6).reshape(2, 3)
        # Populate band 1 data at indices 0, 1, 2, 3
        expected_data_multi_dim[:, :4] = np.arange(6, 14).reshape(2, 4)
        # Populate band 2 data at indices 7, 8
        expected_data_multi_dim[:, 7:9] = np.arange(14, 18).reshape(2, 2)

        expected_data_single_dim = np.empty(9)

        # Populate band 0 data at indices 4, 5, 6
        expected_data_single_dim[4:7] = np.arange(3)
        # Populate band 1 data at indices 0, 1, 2, 3
        expected_data_single_dim[:4] = np.arange(3, 7)
        # Populate band 2 data at indices 7, 8
        expected_data_single_dim[7:9] = np.arange(7, 9)

        all_bands = multi_dim_bands + single_dim_bands

        app = RecombineMeshBands(mesh_mapping_cube, all_bands)
        app.run()
        assert len(app.results) == 2
        single_dim_result = app.results.extract_cube("single_dim_data")
        multi_dim_result = app.results.extract_cube("multi_dim_data")
        np.testing.assert_array_equal(single_dim_result.data, expected_data_single_dim)
        np.testing.assert_array_equal(multi_dim_result.data, expected_data_multi_dim)

    def test_recombine_metadata(
        self, mesh_mapping_cube, multi_dim_bands, single_dim_bands
    ):
        """Test that the recombined result inherits metadata from the bands.

        The band number should be discarded.
        """
        expected_metadata_multi_dim = multi_dim_bands[0].copy().metadata
        expected_metadata_multi_dim.attributes.pop("band_number")

        expected_metadata_single_dim = single_dim_bands[0].copy().metadata
        expected_metadata_single_dim.attributes.pop("band_number")

        all_bands = multi_dim_bands + single_dim_bands

        app = RecombineMeshBands(mesh_mapping_cube, all_bands)
        app.run()
        single_dim_result = app.results.extract_cube("single_dim_data")
        multi_dim_result = app.results.extract_cube("multi_dim_data")
        assert multi_dim_result.metadata == expected_metadata_multi_dim
        assert single_dim_result.metadata == expected_metadata_single_dim
