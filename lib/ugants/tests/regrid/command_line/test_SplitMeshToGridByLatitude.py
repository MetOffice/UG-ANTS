# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from unittest import mock

import numpy as np
import pytest

from ugants.regrid.applications import SplitMeshToGridByLatitude
from ugants.tests.stock import cubedsphere_cube, regular_grid_global_cube
from ugants.utils.cube import as_cubelist


@pytest.fixture()
def sample_cubelist():
    cube = cubedsphere_cube(4)
    return as_cubelist(cube)


@pytest.fixture()
def target_cubelist():
    cube = regular_grid_global_cube(144, 192)
    return as_cubelist(cube)


def _latitude_range(cube):
    latitude = cube.coord(axis="y")
    try:
        latitude.guess_bounds()
    except ValueError:
        # Implies not a dim coord i.e. a Mesh
        pass
    return np.max(latitude.bounds) - np.min(latitude.bounds)


class TestRun:
    def test_number_of_source_bands(self, sample_cubelist, target_cubelist):
        split = SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 2)
        split.run()
        actual = split.source_bands

        assert len(actual) == 2

    def test_source_bands_each_cover_equal_latitude(
        self, sample_cubelist, target_cubelist
    ):
        split = SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 2)
        split.run()
        source_bands = split.source_bands

        assert _latitude_range(source_bands[0]) == _latitude_range(source_bands[1])

    def test_source_bands_are_different(self, sample_cubelist, target_cubelist):
        split = SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 2)
        split.run()
        source_bands = split.source_bands

        assert source_bands[0] != source_bands[1]

    def test_number_of_target_bands(self, sample_cubelist, target_cubelist):
        split = SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 2)
        split.run()
        actual = split.target_bands

        assert len(actual) == 2

    def test_target_bands_each_cover_equal_latitude(
        self, sample_cubelist, target_cubelist
    ):
        split = SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 2)
        split.run()
        target_bands = split.target_bands

        assert _latitude_range(target_bands[0]) == _latitude_range(target_bands[1])

    def test_target_bands_are_different(self, sample_cubelist, target_cubelist):
        split = SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 2)
        split.run()
        target_bands = split.target_bands

        assert target_bands[0] != target_bands[1]

    def test_validate_source(self, sample_cubelist, target_cubelist):
        two_sources = sample_cubelist * 2

        with pytest.raises(ValueError, match="Source contained 2 cubes, expected 1."):
            SplitMeshToGridByLatitude(two_sources, target_cubelist, 2)

    def test_validate_target(self, sample_cubelist, target_cubelist):
        two_targets = target_cubelist * 2

        with pytest.raises(ValueError, match="Target contained 2 cubes, expected 1."):
            SplitMeshToGridByLatitude(sample_cubelist, two_targets, 2)

    def test_validate_number_of_bands(self, sample_cubelist, target_cubelist):
        with pytest.raises(
            ValueError, match="The number of bands must be greater than 1, got 1."
        ):
            SplitMeshToGridByLatitude(sample_cubelist, target_cubelist, 1)


class TestSave:
    def test_save_ugrid(self):
        expected = [
            mock.call("1", "baz/source_band_0.nc"),
            mock.call("2", "baz/source_band_1.nc"),
        ]

        split_mesh = SplitMeshToGridByLatitude(
            ["foo"],
            ["bar"],
            2,
        )
        split_mesh.source_bands = ["1", "2"]
        split_mesh.target_bands = ["3", "4"]
        split_mesh.output = "baz"
        with (
            mock.patch("ugants.regrid.applications.iris.fileformats.netcdf.save"),
            mock.patch("ugants.regrid.applications.save.ugrid") as mock_ugrid,
        ):
            split_mesh.save()
        assert mock_ugrid.call_args_list == expected

    def test_save_netcdf(self):
        expected = [
            mock.call("3", "baz/target_band_0.nc"),
            mock.call("4", "baz/target_band_1.nc"),
        ]

        split_mesh = SplitMeshToGridByLatitude(
            ["foo"],
            ["bar"],
            2,
        )
        split_mesh.source_bands = ["1", "2"]
        split_mesh.target_bands = ["3", "4"]
        split_mesh.output = "baz"
        with (
            mock.patch(
                "ugants.regrid.applications.iris.fileformats.netcdf.save"
            ) as mock_netcdf,
            mock.patch("ugants.regrid.applications.save.ugrid"),
        ):
            split_mesh.save()
        assert mock_netcdf.call_args_list == expected
