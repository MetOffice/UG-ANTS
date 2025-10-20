#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the split -> regrid -> recombine pipeline."""
import pytest
from iris.cube import CubeList

from ugants.io import load
from ugants.regrid.applications import (
    MeshToGridRegrid,
    RecombineGridBands,
    SplitMeshToGridByLatitude,
)
from ugants.tests import get_data_path


def _standard_regrid(source, target_grid, scheme):
    """Regrid from source to target in one go."""
    regrid = MeshToGridRegrid(source, target_grid, scheme)
    regrid.run()
    return regrid.results


def _split_regrid(source, target_grid, scheme, n_bands=2):
    """Regrid from source to target by splitting into latitude bands."""
    split_app = SplitMeshToGridByLatitude(source, target_grid, n_bands)
    split_app.run()

    regrid_bands = CubeList()
    for source_band, target_band in zip(
        split_app.source_bands, split_app.target_bands, strict=True
    ):
        regrid_band = _standard_regrid(
            CubeList([source_band]),
            CubeList([target_band]),
            scheme,
        )
        regrid_bands.append(regrid_band)

    recombine = RecombineGridBands(regrid_bands)
    recombine.run()

    return recombine.results


@pytest.mark.parametrize("n_bands", [2, 3, 4])
class TestConsistentResults:
    """Test that the split regrid results are consistent with regridding in one go."""

    @pytest.mark.parametrize("scheme", ["conservative", "bilinear", "nearest"])
    def test_no_tolerance(self, scheme, n_bands):
        """Test that results are consistent when no tolerance is provided."""
        sources = load.ugrid(get_data_path("data_C4.nc"), "sample_data")
        for source in sources:
            source.attributes["history"] = "foo"
        target_grid = load.cf(get_data_path("non_ugrid_data.nc"))

        expected = _standard_regrid(sources, target_grid, scheme)
        actual = _split_regrid(sources, target_grid, scheme, n_bands=n_bands)

        assert expected == actual
