# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from iris import load_cube
from ugants.tests import get_data_path
from ugants.tests.stock import mesh_cube
from ugants.utils.cube import is_ugrid


def test_lat_lon__false():
    # Use iris.load_cube here rather than ugants.io.load.cf
    # because the ugants implementation calls is_ugrid on load.
    lat_lon_filepath = get_data_path("non_ugrid_data.nc")
    lat_lon_cube = load_cube(lat_lon_filepath)
    actual = is_ugrid(lat_lon_cube)
    assert actual is False


def test_ugrid__true():
    ugrid_cube = mesh_cube()
    actual = is_ugrid(ugrid_cube)
    assert actual is True
