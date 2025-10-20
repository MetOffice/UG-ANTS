# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from tempfile import NamedTemporaryFile

import iris.exceptions
import pytest
from iris import Constraint, load_cube
from iris.coords import DimCoord
from iris.cube import CubeList

from ugants.io import save
from ugants.io.load import ugrid
from ugants.tests import get_data_path


def tests_no_datum_warning():
    """Tests that an iris datum warning is raised when data is loaded.

    Test for future behaviour: #2586
    When this test fails, remove the iris datum warning filter.
    """
    datum_warning = "Ignoring a datum in netCDF load for consistency with existing behaviour. In a future version of Iris, this datum will be applied. To apply the datum when loading, use the iris.FUTURE.datum_support flag."  # noqa: E501
    with pytest.warns(FutureWarning, match=datum_warning) as warnings_given:
        load_cube(get_data_path("non_ugrid_data.nc"))
        if not warnings_given:
            pytest.fail(
                "Remove iris datum warning filter https://code.metoffice.gov.uk/trac/ancil/ticket/2586"
            )


def test_ugrid_sample_load():
    ugrid_sample = ugrid(get_data_path("data_C4.nc"))
    assert isinstance(ugrid_sample, CubeList)
    assert len(ugrid_sample) == 2


def test_non_ugrid_sample_load():
    with pytest.raises(
        iris.exceptions.InvalidCubeError,
        match="Specified file '.*/non_ugrid_data.nc' does not contain UGrid data",
    ):
        ugrid(get_data_path("non_ugrid_data.nc"))


def test_empty_mesh_sample_load():
    with pytest.raises(
        iris.exceptions.InvalidCubeError,
        match=r"No data found in file\(s\) '.*/mesh_C12.nc' matching the "
        r"provided constraint\(s\) 'None'.",
    ):
        ugrid(get_data_path("mesh_C12.nc"))


def test_constrained_on_mesh_coord():
    def greater_than_zero(cell):
        return cell > 0

    latitude_constraint = Constraint(latitude=greater_than_zero)
    with pytest.raises(
        iris.exceptions.InvalidCubeError,
        match=r"Attempting to load UGrid data from '.*/data_C4.nc' with constraint\(s\)"
        r" '.*' has resulted in non-UGrid data being loaded. "
        r"This may be caused by constraining on an unstructured dimension.",
    ):
        ugrid(get_data_path("data_C4.nc"), constraints=latitude_constraint)


def test_ugrid_load_name_constraint():
    ugrid_sample_cubelist = ugrid(get_data_path("data_C4.nc"), "sample_data")
    assert isinstance(ugrid_sample_cubelist, CubeList)
    assert len(ugrid_sample_cubelist) == 1
    ugrid_sample_cube = ugrid_sample_cubelist[0]
    assert ugrid_sample_cube.name() == "sample_data"


def test_cube_not_found():
    with pytest.raises(
        iris.exceptions.InvalidCubeError,
        match=r"No data found in file\(s\) '.*/data_C4.nc' matching the "
        r"provided constraint\(s\) 'no_data_constraint'.",
    ):
        ugrid(get_data_path("data_C4.nc"), constraints="no_data_constraint")


def test_ugrid_load_coordinate_constraint():
    sample_cube = ugrid(get_data_path("data_C4.nc"), "sample_data")[0]
    sample_cube_second_level = sample_cube.copy()
    vertical_levels = DimCoord([0, 1], "height")
    sample_cube.add_aux_coord(vertical_levels[0])
    sample_cube_second_level.add_aux_coord(vertical_levels[1])
    multi_dim_cube = CubeList([sample_cube, sample_cube_second_level]).merge()[0]
    temporary_file = NamedTemporaryFile(suffix=".nc")
    save.ugrid(multi_dim_cube, temporary_file.name)

    height_constraint = Constraint(height=1)
    constrained_loaded_cubelist = ugrid(
        temporary_file.name, constraints=height_constraint
    )
    assert isinstance(constrained_loaded_cubelist, CubeList)
    assert len(constrained_loaded_cubelist) == 1
    constrained_loaded_cube = constrained_loaded_cubelist[0]
    assert constrained_loaded_cube.coord("height").points == vertical_levels[1].points
