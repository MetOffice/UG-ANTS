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
from iris.cube import Cube
from iris.exceptions import ConstraintMismatchError
from numpy.testing import assert_array_equal

from ugants.io import load
from ugants.mask.command_line import GenerateMask, derive_mask
from ugants.tests import get_data_path

OUTPUT_PATH = "/path/to/output.nc"


@pytest.fixture()
def source_path():
    return get_data_path("data_C4.nc")


@pytest.fixture()
def sample_cubelist(source_path):
    """Return a two-element CubeList of C4 UGrid data."""
    return load.ugrid(source_path)


class TestDeriveMask:
    test_cube = Cube([0, 0, 0.3, 0.5, 1, 1])

    def test_derive_land(self):
        land_mask = derive_mask(self.test_cube, "land")
        expected_output = np.array([0, 0, 1, 1, 1, 1], dtype="int8")
        assert_array_equal(land_mask.data, expected_output)

    def test_derive_sea(self):
        sea_mask = derive_mask(self.test_cube, "sea")
        expected_output = np.array([1, 1, 1, 1, 0, 0], dtype="int8")
        assert_array_equal(sea_mask.data, expected_output)

    def test_valid_mask_type(self):
        message_regex = re.compile(
            r"^Provided mask_type 'foo' is not valid. Acceptable values"
            r" are either 'land' or 'sea'"
        )

        with pytest.raises(ValueError) as excinfo:
            derive_mask(self.test_cube, "foo")

        assert message_regex.search(str(excinfo.value))


class TestCLI:
    @pytest.fixture()
    def default_command(self, source_path):
        """Return default command line arguments for the mask application."""
        return [source_path, OUTPUT_PATH, "--mask-type", "sea"]

    @pytest.fixture()
    def default_app(self, default_command):
        return GenerateMask.from_command_line(default_command)

    def test_source_loaded(self, default_app, sample_cubelist):
        assert default_app.land_fraction == sample_cubelist

    def test_output_path_added(self, default_app):
        assert default_app.output == OUTPUT_PATH

    def test_mask_type_added(self, default_app):
        assert default_app.mask_type == "sea"

    def test_invalid_mask_type_fails(self, default_command):
        command = default_command
        command[-1] = "invalid_mask"
        with redirect_stderr(StringIO()) as buffer, pytest.raises(SystemExit):
            GenerateMask.from_command_line(command)
        actual_stderr = buffer.getvalue()
        expected_stderr = re.compile(
            "error: argument --mask-type: invalid choice: "
            r"'invalid_mask' \(choose from 'land', 'sea'\)"
        )
        assert expected_stderr.search(actual_stderr)


class TestGenerateMask:
    def test_no_land_fraction_fails(self, sample_cubelist):
        app = GenerateMask(sample_cubelist, "sea")
        with pytest.raises(
            ConstraintMismatchError,
            match=r"Got 0 cubes for constraint Constraint\(name='land_area_fraction'\),"
            " expecting 1.",
        ):
            app.run()

    def test_derive_mask_call(self, sample_cubelist):
        sample_cubelist[1].standard_name = "land_area_fraction"
        app = GenerateMask(sample_cubelist, "land")
        derive_mask_target = "ugants.mask.command_line.derive_mask"
        with mock.patch(derive_mask_target, autospec=True) as mock_derive_mask:
            app.run()
        mock_derive_mask.assert_called_once_with(
            land_fraction=sample_cubelist[1], mask_type="land"
        )
