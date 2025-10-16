# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import os.path
import re

import pytest
from iris.cube import CubeList
from iris_sample_data import path as sample_data_path

from ugants.io.load import cf


class TestLoadRegular:
    def test_load_netcdf_regular(self):
        input_filepath = os.path.join(sample_data_path, "A1B_north_america.nc")
        cube_list = cf(input_filepath)

        assert isinstance(cube_list, CubeList)

    def test_load_netcdf_mesh(self):
        input_filepath = os.path.join(sample_data_path, "mesh_C4_synthetic_float.nc")
        message_regex = re.compile(r"^Cannot load input file .* as 'cf_role' is .*")
        with pytest.raises(ValueError) as excinfo:
            cf(input_filepath)

        assert message_regex.search(str(excinfo.value))

    def test_load_pp_regular(self):
        input_filepath = os.path.join(sample_data_path, "air_temp.pp")
        message_regex = re.compile(r"Input file .* must be netCDF.")
        with pytest.raises(ValueError) as excinfo:
            cf(input_filepath)

        assert message_regex.search(str(excinfo.value))
