# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for the :func:`ugants.io.save.ugrid` function."""

import unittest.mock

import netCDF4
import pytest
from iris.cube import Cube

from ugants.exceptions import ProvisionalWarning
from ugants.io.save import mesh as save_mesh
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
        testmesh = sample_mesh()
        assert not testpath.exists()
        save_mesh(testmesh, testpath)
        assert testpath.exists()

    def test_provisional_warning(self, testpath):
        testmesh = sample_mesh()
        with pytest.warns(ProvisionalWarning) as warnings:
            save_mesh(testmesh, testpath)

        found_warning = False
        # There's one warning from ug-ants, and at least one warning from
        # iris.  We only care that our warning is included in the list of
        # warnings - we don't care about other warnings, or whether our
        # warning is first, last or somewhere in the middle:
        for warning in warnings:
            if str(warning.message) == PROVISIONAL_WARNING_MESSAGE:
                found_warning = True
        assert found_warning

    @pytest.mark.xfail(reason="Cannot copy a mesh in Iris 3.9")
    def test_provisional_attribute_on_mesh(self, testpath):
        """Test that the mesh is not modified inplace on save."""
        testmesh = sample_mesh()
        assert "ugants_status" not in testmesh.attributes
        save_mesh(testmesh, testpath)
        assert "ugants_status" not in testmesh.attributes

    @pytest.mark.xfail(reason="Cannot add a global attribute to a mesh in Iris 3.9")
    def test_global_provisional_attribute_on_disk(self, testpath):
        """Test that the global attribute has been added to the mesh on disk."""
        testmesh = sample_mesh()
        assert "ugants_status" not in testmesh.attributes
        save_mesh(testmesh, testpath)
        actual = _get_netcdf_global_attribute(testpath, "ugants_status")
        assert actual == PROVISIONAL_WARNING_MESSAGE

    def test_local_provisional_attribute_on_disk(self, testpath):
        """Test that the local attribute has been added to the mesh on disk.

        At Iris 3.9 there is no way of setting global attributes on a mesh,
        so the attributes are set on a variable, i.e. locally.
        """
        testmesh = sample_mesh()
        testmesh.var_name = "dynamics"
        assert "ugants_status" not in testmesh.attributes
        save_mesh(testmesh, testpath)
        actual = _get_netcdf_local_attribute(
            testpath, testmesh.var_name, "ugants_status"
        )
        assert actual == PROVISIONAL_WARNING_MESSAGE

    @pytest.mark.xfail(reason="Cannot copy a mesh in Iris 3.9")
    def test_suite_provenance_attribute_on_cube(self, testpath, monkeypatch):
        """Test that the mesh is not modified inplace on save."""
        testmesh = sample_mesh()
        assert "suite_provenance" not in testmesh.attributes
        attribute_value = "r1234_branch@5678"
        monkeypatch.setenv("SUITE_PROVENANCE", attribute_value)
        save_mesh(testmesh, testpath)
        assert "suite_provenance" not in testmesh.attributes

    @pytest.mark.xfail(reason="Cannot add a global attribute to a mesh in Iris 3.9")
    def test_global_suite_provenance_attribute_on_disk(self, testpath, monkeypatch):
        """Test that the global attribute has been added to the mesh on disk."""
        expected = "r1234_branch@5678"

        testmesh = sample_mesh_cube()
        assert "suite_provenance" not in testmesh.attributes
        monkeypatch.setenv("SUITE_PROVENANCE", expected)
        save_mesh(testmesh, testpath)
        actual = _get_netcdf_global_attribute(testpath, "suite_provenance")

        assert actual == expected

    def test_local_suite_provenance_attribute_on_disk(self, testpath, monkeypatch):
        """Test that the local attribute has been added to the mesh on disk.

        At Iris 3.9 there is no way of setting global attributes on a mesh,
        so the attributes are set on a variable, i.e. locally.
        """
        expected = "r1234_branch@5678"

        testmesh = sample_mesh()
        testmesh.var_name = "dynamics"
        assert "suite_provenance" not in testmesh.attributes
        monkeypatch.setenv("SUITE_PROVENANCE", expected)
        save_mesh(testmesh, testpath)
        actual = _get_netcdf_local_attribute(
            testpath, testmesh.var_name, "suite_provenance"
        )

        assert actual == expected

    @pytest.mark.xfail(reason="Cannot copy a mesh in Iris 3.9")
    def test_none_suite_provenance_on_cube(self, testpath, monkeypatch):
        """Test that the mesh is not modified inplace on save."""
        testmesh = sample_mesh()
        assert "suite_provenance" not in testmesh.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        save_mesh(testmesh, testpath)
        assert "suite_provenance" not in testmesh.attributes

    @pytest.mark.xfail(reason="Cannot add a global attribute to a mesh in Iris 3.9")
    def test_global_none_suite_provenance_on_disk(self, testpath, monkeypatch):
        """Test that the global attribute has been added to the mesh on disk."""
        expected = "None"

        testmesh = sample_mesh()
        assert "suite_provenance" not in testmesh.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        save_mesh(testmesh, testpath)
        actual = _get_netcdf_global_attribute(testpath, "suite_provenance")

        assert actual == expected

    def test_local_none_suite_provenance_on_disk(self, testpath, monkeypatch):
        """Test that the local attribute has been added to the mesh on disk.

        At Iris 3.9 there is no way of setting global attributes on a mesh,
        so the attributes are set on a variable, i.e. locally.
        """
        expected = "None"

        testmesh = sample_mesh()
        testmesh.var_name = "dynamics"
        assert "suite_provenance" not in testmesh.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        save_mesh(testmesh, testpath)
        actual = _get_netcdf_local_attribute(
            testpath, testmesh.var_name, "suite_provenance"
        )

        assert actual == expected

    @pytest.mark.xfail(reason="Cannot copy a mesh in Iris 3.9")
    def test_history_attribute_on_cube(self, testpath):
        """Test that the mesh is not modified inplace on save."""
        testmesh = sample_mesh()
        assert "history" not in testmesh.attributes
        save_mesh(testmesh, testpath)
        assert "history" not in testmesh.attributes

    @pytest.mark.xfail(reason="Cannot add a global attribute to a mesh in Iris 3.9")
    def test_global_history_attribute_on_disk(self, testpath):
        """Test that the global attribute has been added to the mesh on disk."""
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar"

        testmesh = sample_mesh()
        assert "history" not in testmesh.attributes
        with (
            unittest.mock.patch(
                "ugants.utils.cube.datetime", _MockDateTime(reference_date)
            ),
            unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]),
        ):
            save_mesh(testmesh, testpath)
            actual = _get_netcdf_global_attribute(testpath, "history")

        assert actual == expected

    def test_local_history_attribute_on_disk(self, testpath):
        """Test that the local attribute has been added to the mesh on disk.

        At Iris 3.9 there is no way of setting global attributes on a mesh,
        so the attributes are set on a variable, i.e. locally.
        """
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar"

        testmesh = sample_mesh()
        testmesh.var_name = "dynamics"
        assert "history" not in testmesh.attributes
        with (
            unittest.mock.patch(
                "ugants.utils.cube.datetime", _MockDateTime(reference_date)
            ),
            unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]),
        ):
            save_mesh(testmesh, testpath)
            actual = _get_netcdf_local_attribute(testpath, testmesh.var_name, "history")

        assert actual == expected

    @pytest.mark.xfail(reason="Cannot copy a mesh in Iris 3.9")
    def test_extend_history_attribute_on_cube(self, testpath):
        """Test that the mesh is not modified inplace on save.

        Mesh already has history.
        """
        testmesh = sample_mesh()
        testmesh.attributes["history"] = "initial history"
        save_mesh(testmesh, testpath)
        assert testmesh.attributes["history"] == "initial history"

    @pytest.mark.xfail(reason="Cannot add a global attribute to a mesh in Iris 3.9")
    def test_extend_global_history_attribute_on_disk(self, testpath):
        """Test that the global attribute has been added to the mesh on disk.

        Mesh already has history.
        """
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar\ninitial history"

        testmesh = sample_mesh()
        testmesh.attributes["history"] = "initial history"

        with (
            unittest.mock.patch(
                "ugants.utils.cube.datetime", _MockDateTime(reference_date)
            ),
            unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]),
        ):
            save_mesh(testmesh, testpath)
            actual = _get_netcdf_global_attribute(testpath, "history")

        assert actual == expected

    def test_extend_local_history_attribute_on_disk(self, testpath):
        """Test that the local attribute has been added to the mesh on disk.

        At Iris 3.9 there is no way of setting global attributes on a mesh,
        so the attributes are set on a variable, i.e. locally.

        Mesh already has history.
        """
        reference_date = "1970-01-01"
        expected = f"{reference_date}: foo bar\ninitial history"

        testmesh = sample_mesh()
        testmesh.var_name = "dynamics"
        testmesh.attributes["history"] = "initial history"

        with (
            unittest.mock.patch(
                "ugants.utils.cube.datetime", _MockDateTime(reference_date)
            ),
            unittest.mock.patch("ugants.utils.cube.sys.argv", ["foo", "bar"]),
        ):
            save_mesh(testmesh, testpath)
            actual = _get_netcdf_local_attribute(testpath, testmesh.var_name, "history")

        assert actual == expected

    def test_non_string_history(self, testpath):
        """Test that an error is raised when the mesh has a non-string history."""
        testmesh = sample_mesh()
        testmesh.var_name = "dynamics"
        testmesh.attributes["history"] = 1
        with pytest.raises(
            TypeError,
            match="Mesh 'dynamics' has a 'history' attribute of non-string type : 1.",
        ):
            save_mesh(testmesh, testpath)


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


class TestCubes:
    def test_structured_fails(self, testpath):
        """Test that an error is raised for non-mesh cubes."""
        cube = Cube([0])
        with pytest.raises(
            TypeError, match="Expected mesh, got <class 'iris.cube.Cube'>."
        ):
            save_mesh(cube, testpath)

    def test_unstructured_fails(self, testpath):
        """Test that an error is raised for UGrid cubes."""
        cube = sample_mesh_cube()
        with pytest.raises(
            TypeError, match="Expected mesh, got <class 'iris.cube.Cube'>."
        ):
            save_mesh(cube, testpath)


class TestFilepaths:
    def test_string_path(self, testpath):
        """A string should work just as well as a Path object."""
        testmesh = sample_mesh()
        save_mesh(testmesh, str(testpath))
        assert testpath.exists()

    def test_no_extension__fails(self, temporary_filepaths_function):
        testmesh = sample_mesh()
        filepath = temporary_filepaths_function(prefix="temp", suffix="")
        message = (
            f'Provided output filepath, "{filepath!s}", '
            'does not have extension ".nc"'
        )
        with pytest.raises(ValueError, match=message):
            save_mesh(testmesh, filepath)

    def test_bad_extension__fails(self, temporary_filepaths_function):
        testmesh = sample_mesh()
        filepath = temporary_filepaths_function(prefix="temp", suffix=".x23")
        message = (
            f'Provided output filepath, "{filepath!s}", '
            'does not have extension ".nc"'
        )
        with pytest.raises(ValueError, match=message):
            save_mesh(testmesh, filepath)

    def test_multiple_paths__fails(self, temporary_filepaths_function):
        testmesh = sample_mesh()
        filepath1 = temporary_filepaths_function(suffix="*.nc")
        filepath2 = temporary_filepaths_function(suffix="*.nc")
        assert filepath1 != filepath2
        multiple_paths = [filepath1, filepath2]
        message = "expected str, bytes or os.PathLike object, not list"
        with pytest.raises(TypeError, match=message):
            save_mesh(testmesh, multiple_paths)

    def test_unpathlike__fails(self):
        # Effectively, the same error as list-of-paths.
        testmesh = sample_mesh()
        message = "expected str, bytes or os.PathLike object, not NoneType"
        with pytest.raises(TypeError, match=message):
            save_mesh(testmesh, None)


class TestIrisCall:
    def test_iris_save_arguments(self, testpath):
        r"""
        Check that filepath and \*\*save_kwargs are passed down to the iris save call.

        NOTE: In this case, it's simpler to apply a "mockist" style than to come up
        with some suitable, valid, specific usage of the kwargs argument.
        """
        testmesh = sample_mesh()
        # Construct some sample meaningless kwargs.  N.B. we can't provide some
        # abstract mock-type object as kwargs, since the Python "**" call mechanism
        # itself insists on it being a mapping with string keys.
        test_kwargs = {
            "kwarg_a1": unittest.mock.sentinel.a1_value,
            "kwarg_b2": unittest.mock.sentinel.b2_value,
        }
        with unittest.mock.patch("ugants.io.save.save_mesh") as call_patch:
            save_mesh(testmesh, testpath, **test_kwargs)
        # Mesh(es) being saved are not the original meshes, but edited copies.
        # Those edits are captured in other tests here - rather than have
        # redundant additional testing, will allow ANY as first argument for
        # save call:
        assert call_patch.call_args_list == [
            unittest.mock.call(unittest.mock.ANY, testpath, **test_kwargs),
        ]
