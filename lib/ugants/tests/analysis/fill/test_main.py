# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from unittest import mock

import numpy as np
import pytest
from ugants.analysis.command_line import FillMissingPoints
from ugants.io import load
from ugants.tests import get_data_path
from ugants.utils.cube import as_cubelist

OUTPUT_PATH = "/path/to/output.nc"


@pytest.fixture()
def source_path():
    return get_data_path("data_C4.nc")


@pytest.fixture()
def sample_cubelist(source_path):
    """Return a two-element CubeList of C4 UGrid data."""
    return load.ugrid(source_path)


@pytest.fixture()
def source_cubelist(sample_cubelist):
    """Return a single-element CubeList of C4 UGrid data."""
    return as_cubelist(sample_cubelist.extract_cube("sample_data"))


@pytest.fixture()
def target_mask_cubelist(source_cubelist):
    """Return a single-element CubeList of C4 UGrid data."""
    source_cube = source_cubelist[0]
    target_mask = source_cube.copy(np.zeros(source_cube.shape, dtype=int))
    target_mask.data[0] = 1
    target_mask.rename("target_mask")
    return as_cubelist(target_mask)


@pytest.fixture()
def default_command(source_path):
    """Return default command line arguments for the fill application."""
    return [
        source_path,
        OUTPUT_PATH,
    ]


class TestCLINoTargetMask:
    def test_source_loaded(self, default_command, sample_cubelist):
        app = FillMissingPoints.from_command_line(default_command)
        assert app.source == sample_cubelist

    def test_output_path_added(self, default_command):
        app = FillMissingPoints.from_command_line(default_command)
        assert app.output == OUTPUT_PATH

    def test_target_mask_none(self, default_command):
        app = FillMissingPoints.from_command_line(default_command)
        assert app.target_mask is None


class TestCLITargetMask:
    @pytest.fixture()
    def command(self, default_command, source_path):
        return [*default_command, "--target-mask", source_path]

    def test_source_loaded(self, command, sample_cubelist):
        app = FillMissingPoints.from_command_line(command)
        assert app.source == sample_cubelist

    def test_output_path_added(self, command):
        app = FillMissingPoints.from_command_line(command)
        assert app.output == OUTPUT_PATH

    def test_target_mask_loaded(self, command, sample_cubelist):
        app = FillMissingPoints.from_command_line(command)
        assert app.target_mask == sample_cubelist


class TestNoTargetMask:
    @pytest.fixture()
    def default_app(self, source_cubelist):
        return FillMissingPoints(source_cubelist)

    def test_kdtreefill_instantiation(self, default_app, source_cubelist):
        kdtreefill_target = "ugants.analysis.command_line.KDTreeFill"
        with mock.patch(kdtreefill_target) as mock_kdtreefill:
            default_app.run()
        mock_kdtreefill.assert_called_once_with(source_cubelist[0], None)

    @pytest.mark.filterwarnings(
        (
            "ignore:Assuming a spherical geocentric coordinate system for conversion "
            "to 3D cartesian coordinates. If the provided cube is not defined on this "
            "coordinate system then unexpected results may "
            "occur.:UserWarning:ugants.analysis"
        ),
        (
            "ignore:No cells in the source cube require "
            "filling.:UserWarning:ugants.analysis"
        ),
    )
    def test_fill_call(self, default_app, source_cubelist):
        with mock.patch("ugants.analysis.command_line.fill_cube") as mock_fill:
            default_app.run()
        mock_fill.assert_called_once_with(source_cubelist[0], None)


class TestTargetMask:
    @pytest.fixture()
    def default_app(self, source_cubelist, target_mask_cubelist):
        return FillMissingPoints(source_cubelist, target_mask=target_mask_cubelist)

    def test_kdtreefill_instantiation(
        self, default_app, source_cubelist, target_mask_cubelist
    ):
        kdtreefill_target = "ugants.analysis.command_line.KDTreeFill"
        with mock.patch(kdtreefill_target) as mock_kdtreefill:
            default_app.run()
        mock_kdtreefill.assert_called_once_with(
            source_cubelist[0], target_mask_cubelist[0]
        )

    @pytest.mark.filterwarnings(
        (
            "ignore:Assuming a spherical geocentric coordinate system for conversion "
            "to 3D cartesian coordinates. If the provided cube is not defined on this "
            "coordinate system then unexpected results may "
            "occur.:UserWarning:ugants.analysis"
        ),
        (
            "ignore:No cells in the source cube require "
            "filling.:UserWarning:ugants.analysis"
        ),
    )
    def test_fill_call(self, default_app, source_cubelist, target_mask_cubelist):
        with mock.patch("ugants.analysis.command_line.fill_cube") as mock_fill:
            default_app.run()
        mock_fill.assert_called_once_with(source_cubelist[0], target_mask_cubelist[0])

    def test_multi_target_mask_fail(self, source_cubelist, sample_cubelist):
        with pytest.raises(ValueError, match="Expecting one target mask, found 2"):
            FillMissingPoints(source_cubelist, sample_cubelist)
