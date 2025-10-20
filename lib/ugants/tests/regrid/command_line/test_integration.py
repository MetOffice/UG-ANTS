# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the split -> regrid -> recombine pipeline."""
import pytest
from iris.cube import CubeList

from ugants.io import load
from ugants.regrid.command_line import (
    RecombineMeshBands,
    Regrid,
    SplitGridToMeshByLatitude,
)
from ugants.tests import get_data_path


def _standard_regrid(source, target_mesh, scheme, tolerance=0.0):
    """Regrid from source to target in one go."""
    regrid = Regrid(source, target_mesh, scheme, tolerance)
    regrid.run()
    return regrid.results


def _split_regrid(source, target_mesh, scheme, tolerance=0.0, n_bands=2):
    """Regrid from source to target by splitting into latitude bands."""
    split_app = SplitGridToMeshByLatitude(source, target_mesh, n_bands)
    split_app.run()
    mesh_mapping = CubeList([split_app.mesh_mapping_cube])

    regrid_bands = CubeList()
    for source_band, mesh_band in zip(
        split_app.results, split_app.mesh_bands, strict=True
    ):
        regrid_band = _standard_regrid(source_band, mesh_band, scheme, tolerance)
        regrid_bands.extend(regrid_band)

    recombine = RecombineMeshBands(mesh_mapping, regrid_bands)
    recombine.run()

    return recombine.results


@pytest.mark.parametrize("n_bands", [2, 3, 4])
class TestConsistentResultsSingleCube:
    """Test that the split regrid results are consistent with regridding in one go."""

    @pytest.mark.parametrize("scheme", ["conservative", "bilinear", "nearest"])
    def test_no_tolerance(self, scheme, n_bands):
        """Test that results are consistent when no tolerance is provided."""
        source = load.cf(get_data_path("non_ugrid_data.nc")).extract_cube(
            "land_area_fraction"
        )
        target_mesh = load.mesh(get_data_path("mesh_C12.nc"), "dynamics")

        expected = _standard_regrid(source, target_mesh, scheme)
        actual = _split_regrid(source, target_mesh, scheme, n_bands=n_bands)

        assert expected == actual

    @pytest.mark.parametrize("tolerance", [0.2, 0.5, 1.0])
    @pytest.mark.parametrize("scheme", ["conservative", "bilinear"])
    def test_with_tolerance(self, scheme, tolerance, n_bands):
        """Test that results are consistent across tolerances."""
        source = load.cf(get_data_path("non_ugrid_data.nc")).extract_cube(
            "land_area_fraction"
        )
        target_mesh = load.mesh(get_data_path("mesh_C12.nc"), "dynamics")

        expected = _standard_regrid(source, target_mesh, scheme, tolerance)
        actual = _split_regrid(source, target_mesh, scheme, tolerance, n_bands)

        assert expected == actual


@pytest.mark.parametrize("n_bands", [2, 3, 4])
class TestConsistentResultsMultiCube:
    """Test that the split regrid results are consistent with regridding in one go."""

    @pytest.mark.parametrize("scheme", ["conservative", "bilinear", "nearest"])
    def test_no_tolerance(self, scheme, n_bands):
        """Test that results are consistent when no tolerance is provided."""
        source = load.cf(get_data_path("non_ugrid_data.nc"))
        second_source = source[0].copy() + 1
        source.append(second_source)
        target_mesh = load.mesh(get_data_path("mesh_C12.nc"), "dynamics")

        expected = _standard_regrid(source, target_mesh, scheme)
        actual = _split_regrid(source, target_mesh, scheme, n_bands=n_bands)

        # Order of cubes is not guaranteed, so sort by name
        expected.sort(key=lambda cube: cube.name())
        actual.sort(key=lambda cube: cube.name())
        assert expected == actual

    @pytest.mark.parametrize("tolerance", [0.2, 0.5, 1.0])
    @pytest.mark.parametrize("scheme", ["conservative", "bilinear"])
    def test_with_tolerance(self, scheme, tolerance, n_bands):
        """Test that results are consistent across tolerances."""
        source = load.cf(get_data_path("non_ugrid_data.nc"))
        second_source = source[0].copy() + 1
        source.append(second_source)
        target_mesh = load.mesh(get_data_path("mesh_C12.nc"), "dynamics")

        expected = _standard_regrid(source, target_mesh, scheme, tolerance=tolerance)
        actual = _split_regrid(
            source, target_mesh, scheme, n_bands=n_bands, tolerance=tolerance
        )

        # Order of cubes is not guaranteed, so sort by name
        expected.sort(key=lambda cube: cube.name())
        actual.sort(key=lambda cube: cube.name())
        assert expected == actual
