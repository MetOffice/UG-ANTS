# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import re
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

import numpy as np
import pytest
from iris.cube import CubeList

from ugants.io import load
from ugants.regrid.command_line import RegridMeshToMesh
from ugants.tests import get_data_path
from ugants.tests.stock import regular_lat_lon_mesh
from ugants.utils.cube import as_cubelist

OUTPUT_PATH = "/path/to/output.nc"
MESH_NAME = "dynamics"


@pytest.fixture()
def source_path():
    return get_data_path("data_C4.nc")


@pytest.fixture()
def sample_cubelist(source_path):
    """Return a CubeList from a ugrid mesh."""
    return load.ugrid(source_path)


@pytest.fixture()
def single_element_cubelist(sample_cubelist):
    """Return a single-element CubeList from a ugrid mesh."""
    return as_cubelist(sample_cubelist.extract_cube("sample_data"))


@pytest.fixture()
def mesh_path():
    return get_data_path("mesh_C12.nc")


@pytest.fixture()
def mesh_C12(mesh_path):
    return load.mesh(mesh_path, mesh_name=MESH_NAME)


@pytest.fixture()
def synthetic_regular_mesh():
    mesh = regular_lat_lon_mesh()
    return mesh


class TestRegridMeshToMeshCLI:
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
            "--horizontal-regrid-scheme",
            "conservative",
        ]

    @pytest.fixture()
    def default_app(self, default_command):
        return RegridMeshToMesh.from_command_line(default_command)

    def test_source_loaded(self, default_app, sample_cubelist):
        assert default_app.source == sample_cubelist

    def test_output_path_added(self, default_app):
        assert default_app.output == OUTPUT_PATH

    def test_target_mesh_loaded(self, default_app, mesh_C12):
        assert default_app.target_mesh == mesh_C12

    def test_regrid_scheme_added(self, default_app):
        assert default_app.horizontal_regrid_scheme == "conservative"

    def test_default_tolerance_added(self, default_app):
        assert default_app.tolerance == 0.0

    def test_custom_tolerance_added(self, default_command):
        default_command.extend(["--tolerance", "0.5"])
        app = RegridMeshToMesh.from_command_line(default_command)
        assert app.tolerance == 0.5

    def test_invalid_regrid_scheme_fails(self, default_command):
        command = default_command
        command[-1] = "invalid_scheme"
        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            RegridMeshToMesh.from_command_line(command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile(
            "error: argument --horizontal-regrid-scheme: invalid choice: "
            r"'invalid_scheme' \(choose from 'conservative', 'bilinear', 'nearest'\)"
        )
        assert expected_stderr.search(actual_stderr)

    def test_no_mesh_name_given(self, source_path, mesh_path, mesh_C12):
        """Tests that the mesh will be loaded when not supplied with a mesh name."""
        command = [
            source_path,
            OUTPUT_PATH,
            "--target-mesh",
            mesh_path,
            "--horizontal-regrid-scheme",
            "conservative",
        ]
        app = RegridMeshToMesh.from_command_line(command)
        assert app.target_mesh == mesh_C12


class TestRegridMeshToMesh:
    @pytest.fixture()
    def default_app(self, single_element_cubelist, synthetic_regular_mesh):
        return RegridMeshToMesh(
            source=single_element_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme="conservative",
        )

    def test_tolerance_with_nearest_fails(
        self, single_element_cubelist, synthetic_regular_mesh
    ):
        with pytest.raises(
            ValueError,
            match="The 'tolerance' option is not available for regrid scheme 'nearest'",
        ):
            RegridMeshToMesh(
                source=single_element_cubelist,
                target_mesh=synthetic_regular_mesh,
                horizontal_regrid_scheme="nearest",
                tolerance=0.5,
            )

    def test_nearest_regridder_instantiation(
        self, single_element_cubelist, synthetic_regular_mesh
    ):
        """Test that the ESMFNearestRegridder is instantiated with appropriate arguments."""  # noqa: E501
        regridder_target = "ugants.regrid.command_line.esmf_regrid.ESMFNearestRegridder"
        app = RegridMeshToMesh(
            source=single_element_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme="nearest",
        )
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()

        mock_regridder.assert_called_once_with(
            src=single_element_cubelist[0],
            tgt=synthetic_regular_mesh,
            tgt_location="face",
        )

    def test_conservative_regridder_instantiation(
        self, single_element_cubelist, synthetic_regular_mesh
    ):
        """Test that the ESMFAreaWeightedRegridder is instantiated with appropriate arguments."""  # noqa: E501
        regridder_target = (
            "ugants.regrid.command_line.esmf_regrid.ESMFAreaWeightedRegridder"
        )
        app = RegridMeshToMesh(
            source=single_element_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme="conservative",
            tolerance=0.5,
        )
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()

        mock_regridder.assert_called_once_with(
            src=single_element_cubelist[0],
            tgt=synthetic_regular_mesh,
            tgt_location="face",
            mdtol=0.5,
        )

    def test_bilinear_regridder_instantiation(
        self, single_element_cubelist, synthetic_regular_mesh
    ):
        """Test that the ESMFBilinearRegridder is instantiated with appropriate arguments."""  # noqa: E501
        regridder_target = (
            "ugants.regrid.command_line.esmf_regrid.ESMFBilinearRegridder"
        )
        app = RegridMeshToMesh(
            source=single_element_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme="bilinear",
            tolerance=0.5,
        )
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()

        mock_regridder.assert_called_once_with(
            src=single_element_cubelist[0],
            tgt=synthetic_regular_mesh,
            tgt_location="face",
            mdtol=0.5,
        )

    def test_multiple_source_cubes_regridder_calls(
        self, sample_cubelist, synthetic_regular_mesh
    ):
        """Test that the regridder is called with each source cube."""
        regridder_target = (
            "ugants.regrid.command_line.esmf_regrid.ESMFBilinearRegridder"
        )
        app = RegridMeshToMesh(
            source=sample_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme="bilinear",
            tolerance=0.5,
        )
        with mock.patch(regridder_target, autospec=True) as mock_regridder:
            app.run()

        expected_calls = [mock.call(sample_cubelist[0]), mock.call(sample_cubelist[1])]
        # Note we are testing the return_value of the regridder here, to test the
        # arguments passed to the instantiated regridder object, rather than the
        # arguments passed to instantiate the regridder object.
        mock_regridder.return_value.assert_has_calls(expected_calls)


@pytest.mark.parametrize("scheme", ["conservative", "bilinear"])
class TestConsistencyInResults:
    """Unmasked cells in the output should have the same data, regardless of tolerance."""  # noqa: E501

    def test_source_fully_unmasked(
        self, scheme, single_element_cubelist, synthetic_regular_mesh
    ):
        """Results should be identical.

        Since all cells in the source are unmasked, the tolerance option
        should have no effect on the results.
        """
        # Run RegridMeshToMesh without tolerance
        app_without_tolerance = RegridMeshToMesh(
            source=single_element_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme=scheme,
        )
        app_without_tolerance.run()
        regrid_mesh_to_mesh_without_tolerance = app_without_tolerance.results
        # Run RegridMeshToMesh with tolerance
        app_with_tolerance = RegridMeshToMesh(
            source=single_element_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme=scheme,
            tolerance=0.5,
        )
        app_with_tolerance.run()
        regrid_mesh_to_mesh_with_tolerance = app_with_tolerance.results
        assert (
            regrid_mesh_to_mesh_with_tolerance == regrid_mesh_to_mesh_without_tolerance
        )

    def test_source_partially_masked(
        self, scheme, single_element_cubelist, synthetic_regular_mesh
    ):
        """Results in the overlapping unmasked regions should be equivalent.

        Cells that are unmasked in the RegridMeshToMesh results both
        with and without tolerance, should be the same.
        """
        # Mask some cells in the source data.
        source_cube = single_element_cubelist[0].copy()
        source_cube.data[source_cube.data == 1] = np.ma.masked
        source_cubelist = CubeList([source_cube])
        # Run RegridMeshToMesh without tolerance
        app_without_tolerance = RegridMeshToMesh(
            source=source_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme=scheme,
        )
        app_without_tolerance.run()
        regrid_mesh_to_mesh_without_tolerance = app_without_tolerance.results[0]
        # Run RegridMeshToMesh with tolerance
        app_with_tolerance = RegridMeshToMesh(
            source=source_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme=scheme,
            tolerance=0.5,
        )
        app_with_tolerance.run()
        regrid_mesh_to_mesh_with_tolerance = app_with_tolerance.results[0]
        unmasked_in_both = np.logical_and(
            regrid_mesh_to_mesh_without_tolerance.data.mask,
            regrid_mesh_to_mesh_with_tolerance.data.mask,
        )
        np.testing.assert_array_equal(
            regrid_mesh_to_mesh_without_tolerance.data[unmasked_in_both],
            regrid_mesh_to_mesh_with_tolerance.data[unmasked_in_both],
        )

    def test_increased_tolerance_decreases_masked_cells(
        self, scheme, single_element_cubelist, synthetic_regular_mesh
    ):
        """The number of masked cells in the output should be fewer with tolerance than without.

        Consider the example below. The diagram represents a single target cell.
        A small region of the target cell overlaps with
        masked cells in the source (represented by X).
        +-------+
        |       |  If tolerance = 0, then the target cell is masked.
        |     XX|  If tolerance = 0.5, then the target cell is unmasked.
        |     XX|
        +-------+
        """  # noqa: E501
        # Mask some cells in the source data.
        source_cube = single_element_cubelist[0].copy()
        source_cube.data[source_cube.data > 50.0] = np.ma.masked
        source_cubelist = CubeList([source_cube])
        # Run RegridMeshToMesh without tolerance
        app_without_tolerance = RegridMeshToMesh(
            source=source_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme=scheme,
        )
        app_without_tolerance.run()
        regrid_mesh_to_mesh_without_tolerance = app_without_tolerance.results[0]
        mask_without_tolerance = regrid_mesh_to_mesh_without_tolerance.data.mask
        # Run regrid with tolerance
        app_with_tolerance = RegridMeshToMesh(
            source=source_cubelist,
            target_mesh=synthetic_regular_mesh,
            horizontal_regrid_scheme=scheme,
            tolerance=0.5,
        )
        app_with_tolerance.run()
        regrid_mesh_to_mesh_with_tolerance = app_with_tolerance.results[0]
        mask_with_tolerance = regrid_mesh_to_mesh_with_tolerance.data.mask
        assert mask_with_tolerance.sum() < mask_without_tolerance.sum()
