# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the :func:`ugants.io.save.ugrid` function."""
import unittest.mock

import netCDF4
import numpy as np
import pytest
from iris.cube import Cube, CubeList

from ugants.exceptions import ProvisionalWarning
from ugants.io.load import mesh as load_mesh
from ugants.io.load import ugrid as load_ugrid
from ugants.io.save import ugrid as save_ugrid
from ugants.tests import get_data_path
from ugants.tests.stock import sample_mesh, sample_mesh_cube

PROVISIONAL_WARNING_MESSAGE = (
    "The results output by UG-ANTS are provisional and subject to change."
)
# Ignore the provisional output warning message for all tests
pytestmark = pytest.mark.filterwarnings(
    f"ignore:{PROVISIONAL_WARNING_MESSAGE}:ugants.io.save.ProvisionalWarning"
)


# A local derived fixture to generate temporary netcdf filepaths
@pytest.fixture()
def testpath(temporary_filepaths_function):
    path = temporary_filepaths_function(suffix=".nc")
    return path


def _get_netcdf_global_attribute(path, attribute):
    """Get a global attribute from a netCDF file on disk.

    Parameters
    ----------
    path : str
        Filename from which the attribute should be extracted
    attribute : str
        Attribute to extract from the file.

    Returns
    -------
    str or None
        If the attribute is present in the file, then the value of the
        attribute is returned.  Otherwise, return None (which means tests will
        give a test failure message, rather than an attribute not found error).
    """
    on_disk = netCDF4.Dataset(path)
    result = getattr(on_disk, attribute, None)
    on_disk.close()
    return result


def _get_netcdf_local_attribute(path, variable_name, attribute):
    """Get a local attribute from a netCDF file on disk.

    Parameters
    ----------
    path : str
        Filename from which the attribute should be extracted.
    variable_name : str
        Name of the variable from which to retrieve the attribute.
    attribute : str
        Attribute to extract from the file.

    Returns
    -------
    str or None
        If the variable attribute is present in the file, then the value of the
        attribute is returned.  Otherwise, return None (which means tests will
        give a test failure message, rather than an attribute not found error).
        If the variable does not exist in the file, raise an error.

    Raises
    ------
    KeyError
        If the variable does not exist at the path provided.
    """
    on_disk = netCDF4.Dataset(path)
    try:
        variable = on_disk.variables[variable_name]
    except KeyError as exc:
        raise KeyError(f"{variable_name} not found in file ar {path}.") from exc
    result = getattr(variable, attribute, None)
    on_disk.close()
    return result


class TestBasicSaves:
    def test_simple(self, testpath):
        testcube = sample_mesh_cube()
        assert not testpath.exists()
        save_ugrid(testcube, testpath)
        assert testpath.exists()

    def test_provisional_warning(self, testpath):
        testcube = sample_mesh_cube()
        with pytest.warns(ProvisionalWarning) as warnings:
            save_ugrid(testcube, testpath)

        found_warning = False
        # There's one warning from ug-ants, and at least one warning from
        # iris.  We only care that our warning is included in the list of
        # warnings - we don't care about other warnings, or whether our
        # warning is first, last or somewhere in the middle:
        for warning in warnings:
            if str(warning.message) == PROVISIONAL_WARNING_MESSAGE:
                found_warning = True
        assert found_warning

    def test_provisional_attribute_on_cube(self, testpath):
        testcube = sample_mesh_cube()
        assert "ugants_status" not in testcube.attributes
        save_ugrid(testcube, testpath)
        assert "ugants_status" not in testcube.attributes

    def test_provisional_attribute_on_disk(self, testpath):
        testcube = sample_mesh_cube()
        assert "ugants_status" not in testcube.attributes
        save_ugrid(testcube, testpath)
        actual = _get_netcdf_global_attribute(testpath, "ugants_status")
        assert actual == PROVISIONAL_WARNING_MESSAGE

    def test_suite_provenance_attribute_on_cube(self, testpath, monkeypatch):
        testcube = sample_mesh_cube()
        assert "suite_provenance" not in testcube.attributes
        attribute_value = "r1234_branch@5678"
        monkeypatch.setenv("SUITE_PROVENANCE", attribute_value)
        save_ugrid(testcube, testpath)
        assert "suite_provenance" not in testcube.attributes

    def test_suite_provenance_attribute_on_disk(self, testpath, monkeypatch):
        expected = "r1234_branch@5678"

        testcube = sample_mesh_cube()
        assert "suite_provenance" not in testcube.attributes
        monkeypatch.setenv("SUITE_PROVENANCE", expected)
        save_ugrid(testcube, testpath)
        actual = _get_netcdf_global_attribute(testpath, "suite_provenance")

        assert actual == expected

    def test_none_suite_provenance_on_cube(self, testpath, monkeypatch):
        testcube = sample_mesh_cube()
        assert "suite_provenance" not in testcube.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        save_ugrid(testcube, testpath)
        assert "suite_provenance" not in testcube.attributes

    def test_none_suite_provenance_on_disk(self, testpath, monkeypatch):
        expected = "None"

        testcube = sample_mesh_cube()
        assert "suite_provenance" not in testcube.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        save_ugrid(testcube, testpath)
        actual = _get_netcdf_global_attribute(testpath, "suite_provenance")

        assert actual == expected

    def test_history_attribute_on_cube(self, testpath):
        testcube = sample_mesh_cube()
        assert "history" not in testcube.attributes
        save_ugrid(testcube, testpath)
        assert "history" not in testcube.attributes

    def test_history_attribute_on_disk(self, testpath):
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar"

        testcube = sample_mesh_cube()
        assert "history" not in testcube.attributes
        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(testcube, testpath)
            actual = _get_netcdf_global_attribute(testpath, "history")

        assert actual == expected


class _MockDateTime:
    """Acts as a replacement for datetime to give a fixed date.

    Datetime usage is via Datetime().today().replace().isoformat() - this just
    returns itself for the today and replace methods, and finally returns the
    fixed time for the isoformat method.

    """

    def __init__(self, reference_time="1970-01-01"):
        self._reference_time = reference_time

    def today(self):
        return self

    def replace(self, microsecond):
        return self

    def isoformat(self):
        return self._reference_time


class TestStructuredCubes:
    def test_structured_fails(self, testpath):
        # Check that an error is raised for non-mesh cubes.
        cube = Cube([0])
        with pytest.raises(ValueError) as error:
            save_ugrid(cube, testpath)
        assert str(error.value) == "Provided cubes are not all unstructured."


class TestMultipleCubes:
    def test_multi_cube(self, testpath):
        """Check that we can save multiple cubes with the same mesh."""
        # Create a second cube : ensure they have the *same* mesh
        testcube = sample_mesh_cube()
        testcube2 = sample_mesh_cube(mesh=testcube.mesh)
        testcube2.rename("cube2")  # the two cubes must be distinguishable.
        assert not testpath.exists()
        save_ugrid(CubeList([testcube, testcube2]), testpath)
        assert len(load_ugrid(testpath)) == 2
        assert load_mesh(testpath, "Mesh_2d").name() == "Mesh_2d"

    def test_multi_mesh__fails(self, testpath):
        """Check error for saving data on multiple different meshes."""
        # Create a second cube : they have *difference* meshes
        testcube = sample_mesh_cube()
        cube2 = sample_mesh_cube(mesh=sample_mesh(n_faces=5, n_nodes=3, n_edges=0))
        cube2.rename("cube2")
        with pytest.raises(ValueError) as error:
            save_ugrid(CubeList([testcube, cube2]), testpath)
        assert str(error.value) == "Provided cubes are not all on the same mesh."

    def test_same_mesh_different_locations(self, testpath):
        """Save multiple cubes with the same mesh, different data locations."""
        sample_mesh = load_mesh(get_data_path("mesh_C12.nc"), "dynamics")
        face_mesh_coords_x, face_mesh_coords_y = sample_mesh.to_MeshCoords("face")
        node_mesh_coords_x, node_mesh_coords_y = sample_mesh.to_MeshCoords("node")
        face_cube = Cube(
            standard_name="surface_altitude",
            data=np.ones(face_mesh_coords_x.shape),
            long_name="face_data",
            aux_coords_and_dims=[(face_mesh_coords_x, 0), (face_mesh_coords_y, 0)],
            units="metres",
        )
        node_cube = Cube(
            standard_name="surface_altitude",
            data=np.zeros(node_mesh_coords_x.shape),
            long_name="node_data",
            aux_coords_and_dims=[(node_mesh_coords_x, 0), (node_mesh_coords_y, 0)],
            units="metres",
        )
        assert not testpath.exists()
        save_ugrid(CubeList([face_cube, node_cube]), testpath)

        reloaded = load_ugrid(testpath)
        assert len(reloaded) == 2

        reloaded_face = reloaded.extract_cube("face_data")
        reloaded_node = reloaded.extract_cube("node_data")
        assert reloaded_face.location == "face"
        assert reloaded_face.standard_name == "surface_altitude"
        assert reloaded_node.location == "node"
        assert reloaded_node.standard_name == "surface_altitude"
        assert reloaded_node.mesh == reloaded_face.mesh

    def test_same_mesh_different_fields(self, testpath):
        """Save multiple cubes with the same mesh, same location, different fields."""
        sample_mesh = load_mesh(get_data_path("mesh_C12.nc"), "dynamics")
        face_mesh_coords_x, face_mesh_coords_y = sample_mesh.to_MeshCoords("face")
        cube_1 = Cube(
            standard_name="surface_altitude",
            data=np.ones(face_mesh_coords_x.shape),
            long_name="field_1",
            aux_coords_and_dims=[(face_mesh_coords_x, 0), (face_mesh_coords_y, 0)],
            units="metres",
        )
        cube_2 = Cube(
            standard_name="sea_surface_temperature",
            data=np.zeros(face_mesh_coords_x.shape),
            long_name="field_2",
            aux_coords_and_dims=[(face_mesh_coords_x, 0), (face_mesh_coords_y, 0)],
            units="kelvin",
        )
        assert not testpath.exists()
        save_ugrid(CubeList([cube_1, cube_2]), testpath)

        reloaded = load_ugrid(testpath)
        assert len(reloaded) == 2

        reloaded_1 = reloaded.extract_cube("surface_altitude")
        reloaded_2 = reloaded.extract_cube("sea_surface_temperature")

        assert reloaded_1.location == "face"
        assert reloaded_1.standard_name == "surface_altitude"
        assert reloaded_1.long_name == "field_1"

        assert reloaded_2.location == "face"
        assert reloaded_2.standard_name == "sea_surface_temperature"
        assert reloaded_2.long_name == "field_2"

        assert reloaded_2.mesh == reloaded_1.mesh

    def test_history_attribute_on_cubes(self, testpath):
        """Cubes are not changed in place when saving."""
        testcube_1 = sample_mesh_cube()
        testcube_2 = sample_mesh_cube()
        assert "history" not in testcube_1.attributes
        assert "history" not in testcube_2.attributes
        cubes = CubeList([testcube_1, testcube_2])
        save_ugrid(cubes, testpath)
        assert "history" not in testcube_1.attributes
        assert "history" not in testcube_2.attributes

    def test_neither_cube_has_initial_history__global_on_disk(self, testpath):
        """Global history attribute is added when cubes have no initial history."""
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        assert "history" not in testcube_1.attributes

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        assert "history" not in testcube_2.attributes

        cubes = CubeList([testcube_1, testcube_2])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)
        actual = _get_netcdf_global_attribute(testpath, "history")

        assert actual == expected

    def test_neither_cube_has_initial_history__local_on_disk(self, testpath):
        """No local history attribute is added when cubes have no initial history."""
        reference_date = "1970-01-01"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        assert "history" not in testcube_1.attributes

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        assert "history" not in testcube_2.attributes

        cubes = CubeList([testcube_1, testcube_2])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)

        actual_local_history_1 = _get_netcdf_local_attribute(
            testpath, "testcube_1", "history"
        )
        actual_local_history_2 = _get_netcdf_local_attribute(
            testpath, "testcube_2", "history"
        )
        assert actual_local_history_1 is None
        assert actual_local_history_2 is None

    def test_cubes_have_same_initial_history__global_on_disk(self, testpath):
        """Global history attribute is added when cubes have same history."""
        initial_history = "Common history."
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar\n{initial_history}"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        testcube_1.attributes["history"] = initial_history

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        testcube_2.attributes["history"] = initial_history

        cubes = CubeList([testcube_1, testcube_2])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)
        actual_global_history = _get_netcdf_global_attribute(testpath, "history")

        assert actual_global_history == expected

    def test_cubes_have_same_initial_history__local_on_disk(self, testpath):
        """No local history attribute is added when cubes have same history."""
        initial_history = "Common history."
        reference_date = "1970-01-01"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        testcube_1.attributes["history"] = initial_history

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        testcube_2.attributes["history"] = initial_history

        cubes = CubeList([testcube_1, testcube_2])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)

        actual_local_history_1 = _get_netcdf_local_attribute(
            testpath, "testcube_1", "history"
        )
        actual_local_history_2 = _get_netcdf_local_attribute(
            testpath, "testcube_2", "history"
        )
        assert actual_local_history_1 is None
        assert actual_local_history_2 is None

    def test_cubes_have_different_initial_history__local_on_disk(self, testpath):
        """Local history attribute is added when cubes have different histories."""
        reference_date = "1970-01-01"
        history_to_add = f"{reference_date}: foo bar"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        initial_history_1 = "Cube 1 history."
        testcube_1.attributes["history"] = initial_history_1
        expected_history_1 = f"{history_to_add}\n{initial_history_1}"

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        initial_history_2 = "Cube 2 history."
        testcube_2.attributes["history"] = initial_history_2
        expected_history_2 = f"{history_to_add}\n{initial_history_2}"

        cubes = CubeList([testcube_1, testcube_2])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)

        actual_history_1 = _get_netcdf_local_attribute(
            testpath, "testcube_1", "history"
        )
        actual_history_2 = _get_netcdf_local_attribute(
            testpath, "testcube_2", "history"
        )

        assert actual_history_1 == expected_history_1
        assert actual_history_2 == expected_history_2

    def test_cubes_have_different_initial_history__global_on_disk(self, testpath):
        """No global history attribute is added when cubes have different histories."""
        reference_date = "1970-01-01"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        initial_history_1 = "Cube 1 history."
        testcube_1.attributes["history"] = initial_history_1

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        initial_history_2 = "Cube 2 history."
        testcube_2.attributes["history"] = initial_history_2

        cubes = CubeList([testcube_1, testcube_2])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)

        actual_global_history = _get_netcdf_global_attribute(testpath, "history")

        assert actual_global_history is None

    def test_three_cubes_two_common_history__local_on_disk(self, testpath):
        """Local history attribute added, even though two cubes have common history."""
        common_history = "Common history."
        different_history = "Different history."
        reference_date = "1970-01-01"
        history_to_add = f"{reference_date}: foo bar"
        expected_common_history = f"{history_to_add}\n{common_history}"
        expected_different_history = f"{history_to_add}\n{different_history}"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        testcube_1.attributes["history"] = common_history

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        testcube_2.attributes["history"] = common_history

        testcube_3 = sample_mesh_cube()
        testcube_3.rename("testcube_3")
        testcube_3.attributes["history"] = different_history

        cubes = CubeList([testcube_1, testcube_2, testcube_3])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)

        actual_history_1 = _get_netcdf_local_attribute(
            testpath, "testcube_1", "history"
        )
        actual_history_2 = _get_netcdf_local_attribute(
            testpath, "testcube_2", "history"
        )
        actual_history_3 = _get_netcdf_local_attribute(
            testpath, "testcube_3", "history"
        )

        assert actual_history_1 == expected_common_history
        assert actual_history_2 == expected_common_history
        assert actual_history_3 == expected_different_history

    def test_three_cubes_two_common_history__global_on_disk(self, testpath):
        """No global history attribute added, although two cubes have common history."""
        common_history = "Common history."
        different_history = "Different history."
        reference_date = "1970-01-01"

        testcube_1 = sample_mesh_cube()
        testcube_1.rename("testcube_1")
        testcube_1.attributes["history"] = common_history

        testcube_2 = sample_mesh_cube()
        testcube_2.rename("testcube_2")
        testcube_2.attributes["history"] = common_history

        testcube_3 = sample_mesh_cube()
        testcube_3.rename("testcube_3")
        testcube_3.attributes["history"] = different_history

        cubes = CubeList([testcube_1, testcube_2, testcube_3])

        with unittest.mock.patch(
            "ugants.utils.cube.datetime", _MockDateTime(reference_date)
        ), unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]):
            save_ugrid(cubes, testpath)

        actual_global_history = _get_netcdf_global_attribute(testpath, "history")
        assert actual_global_history is None


class TestFilepaths:
    def test_string_path(self, testpath):
        # A string should work just as well as a Path object.
        testcube = sample_mesh_cube()
        save_ugrid(testcube, str(testpath))
        assert testpath.exists()

    def test_no_extension__fails(self, temporary_filepaths_function):
        testcube = sample_mesh_cube()
        filepath = temporary_filepaths_function(prefix="temp", suffix="")
        with pytest.raises(ValueError) as error:
            save_ugrid(testcube, filepath)
        message = (
            f'Provided output filepath, "{filepath!s}", '
            'does not have extension ".nc"'
        )
        assert str(error.value) == message

    def test_bad_extension__fails(self, temporary_filepaths_function):
        testcube = sample_mesh_cube()
        filepath = temporary_filepaths_function(prefix="temp", suffix=".x23")
        with pytest.raises(ValueError) as error:
            save_ugrid(testcube, filepath)
        message = (
            f'Provided output filepath, "{filepath!s}", '
            'does not have extension ".nc"'
        )
        assert str(error.value) == message

    def test_multiple_paths__fails(self, temporary_filepaths_function):
        testcube = sample_mesh_cube()
        filepath1 = temporary_filepaths_function(suffix="*.nc")
        filepath2 = temporary_filepaths_function(suffix="*.nc")
        assert filepath1 != filepath2
        multiple_paths = [filepath1, filepath2]
        with pytest.raises(TypeError) as error:
            save_ugrid(testcube, multiple_paths)
        message = "expected str, bytes or os.PathLike object, not list"
        assert str(error.value) == message

    def test_unpathlike__fails(self, temporary_filepaths_function):
        # Effectively, the same error as list-of-paths.
        testcube = sample_mesh_cube()
        with pytest.raises(TypeError) as error:
            save_ugrid(testcube, None)
        message = "expected str, bytes or os.PathLike object, not NoneType"
        assert str(error.value) == message


class TestIrisCall:
    def test_iris_save_arguments(self, testpath):
        r"""
        Check that filepath and \*\*save_kwargs are passed down to the iris save call.

        NOTE: In this case, it's simpler to apply a "mockist" style than to come up
        with some suitable, valid, specific usage of the kwargs argument.
        """
        testcube = sample_mesh_cube()
        # Construct some sample meaningless kwargs.  N.B. we can't provide some
        # abstract mock-type object as kwargs, since the Python "**" call mechanism
        # itself insists on it being a mapping with string keys.
        test_kwargs = {
            "kwarg_a1": unittest.mock.sentinel.a1_value,
            "kwarg_b2": unittest.mock.sentinel.b2_value,
        }
        with unittest.mock.patch("ugants.io.save.save_netcdf") as call_patch:
            save_ugrid(testcube, testpath, **test_kwargs)
        # Cube(s) being saved are not the original cubes, but edited copies.
        # Those edits are captured in other tests here - rather than have
        # redundant additional testing, will allow ANY as first argument for
        # save call:
        assert call_patch.call_args_list == [
            unittest.mock.call(unittest.mock.ANY, testpath, **test_kwargs),
        ]
