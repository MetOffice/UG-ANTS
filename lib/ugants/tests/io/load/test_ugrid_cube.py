# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from tempfile import NamedTemporaryFile

import iris.exceptions
import pytest
from iris import Constraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from ugants.io import save
from ugants.io.load import ugrid_cube
from ugants.tests import get_data_path


def test_ugrid_sample_load():
    ugrid_sample = ugrid_cube(get_data_path("data_C4.nc"), "sample_data")
    assert isinstance(ugrid_sample, Cube)


def test_ugrid_cubelist():
    with pytest.raises(
        iris.exceptions.ConstraintMismatchError,
        match="failed to merge into a single cube",
    ):
        ugrid_cube(get_data_path("data_C4.nc"))


def test_non_ugrid_sample_load():
    with pytest.raises(
        iris.exceptions.InvalidCubeError,
        match="Specified file '.*/non_ugrid_data.nc' does not contain UGrid data",
    ):
        ugrid_cube(get_data_path("non_ugrid_data.nc"), "sample_data")


def test_cube_not_found():
    with pytest.raises(
        iris.exceptions.ConstraintMismatchError,
        match="no cubes found",
    ):
        ugrid_cube(get_data_path("data_C4.nc"), constraint="no_data_constraint")


def test_ugrid_load_coordinate_constraint():
    sample_cube = ugrid_cube(get_data_path("data_C4.nc"), "sample_data")
    sample_cube_second_level = sample_cube.copy()
    vertical_levels = DimCoord([0, 1], "height")
    sample_cube.add_aux_coord(vertical_levels[0])
    sample_cube_second_level.add_aux_coord(vertical_levels[1])
    multi_dim_cube = CubeList([sample_cube, sample_cube_second_level]).merge()[0]
    temporary_file = NamedTemporaryFile(suffix=".nc")
    save.ugrid(multi_dim_cube, temporary_file.name)

    height_constraint = Constraint(height=1)
    constrained_loaded_cube = ugrid_cube(
        temporary_file.name, constraint=height_constraint
    )
    assert isinstance(constrained_loaded_cube, Cube)
    assert constrained_loaded_cube.coord("height").points == vertical_levels[1].points
