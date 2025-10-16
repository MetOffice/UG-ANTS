# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from copy import deepcopy
from unittest import mock

import pytest
from iris.cube import Cube

from ugants.filter.generic import UnstructuredFilterABC
from ugants.tests.stock import sample_mesh, sample_mesh_cube


class MyTestFilter(UnstructuredFilterABC):
    """A simple mesh filter implementation for testing."""

    def __init__(self, cube, *create_args, **create_kwargs):
        super().__init__(cube)
        # Store the extra creation control parameters.
        self._create_args = create_args
        self._create_kwargs = create_kwargs
        # Initialise the hook for inner call result checking.
        self._fixed_inner_result = None

    def inner_filter(self, cube, standard_name):
        # Test filter operation :  attach attributes recording the operation details
        cube.attributes["original-name"] = standard_name

        result = cube
        if self._fixed_inner_result is not None:
            # Provide an alternative result value, when enabled
            # NOTE: this is basically an alternative to mock testing
            result = self._fixed_inner_result
        else:
            # Just to prove we can capture + use creation args and kwargs
            result.attributes["_create_args"] = self._create_args
            result.attributes["_create_kwargs"] = self._create_kwargs
        return result

    def fix_result_value_to(self, result_value):
        """Fix the result of the 'inner_filter' call, for validity testing."""
        self._fixed_inner_result = result_value


class TestCreate:
    """Test the filter constructor call."""

    def test_create(self):
        input_cube = sample_mesh_cube()
        filter_instance = MyTestFilter(input_cube)
        assert isinstance(filter_instance, MyTestFilter)

    def test_mesh(self):
        cube = sample_mesh_cube()
        filter_instance = MyTestFilter(cube)
        assert filter_instance.mesh == cube.mesh

    def test_location(self):
        cube = sample_mesh_cube()
        filter_instance = MyTestFilter(cube)
        assert filter_instance.location == cube.location

    def test_additional_args(self):
        cube = sample_mesh_cube()
        extra_args = (mock.sentinel.arg1, mock.sentinel.arg2)
        filter_instance = MyTestFilter(cube, *extra_args)
        assert filter_instance._create_args == extra_args

    def test_additional_kwargs(self):
        cube = sample_mesh_cube()
        extra_kwargs = {"key1": mock.sentinel.kwarg1, "key2": mock.sentinel.kwarg2}
        filter_instance = MyTestFilter(cube, **extra_kwargs)
        assert filter_instance._create_kwargs == extra_kwargs

    def test_noncube__fail(self):
        with pytest.raises(Exception) as error:
            MyTestFilter(None)

        expected = (
            "TypeError("
            "'First argument of UnstructuredFilterABC must be a Cube : got \"None\".'"
            ")"
        )
        assert repr(error.value) == expected

    def test_nonmeshcube__fail(self):
        cube = Cube([0])

        with pytest.raises(Exception) as error:
            MyTestFilter(cube)

        expected = (
            "ValueError("
            '\'Cannot base an UnstructuredFilterABC on cube "unknown", '
            "as it has no mesh.'"
            ")"
        )
        assert repr(error.value) == expected


class TestFilterCall:
    """Check that actual filtering is performed as expected."""

    def test_result_type(self):
        cube = sample_mesh_cube()
        filter_instance = MyTestFilter(cube)

        result = filter_instance(cube)

        assert isinstance(result, Cube)

    def test_standardname_removed(self):
        cube = sample_mesh_cube()
        # Give the cube a valid standard-name
        standard_name = "air_temperature"
        cube.standard_name = standard_name
        filter_instance = MyTestFilter(cube)

        result = filter_instance(cube)

        assert result.standard_name is None

    def test_standardname_passed(self):
        cube = sample_mesh_cube()
        # Give the cube a valid standard-name
        standard_name = "air_temperature"
        cube.standard_name = standard_name
        filter_instance = MyTestFilter(cube)

        result = filter_instance(cube)

        assert result.attributes.get("original-name") == standard_name

    def test_extra_args_passed(self):
        cube = sample_mesh_cube()
        extra_args = (mock.sentinel.arg1, mock.sentinel.arg2)
        filter_instance = MyTestFilter(cube, *extra_args)

        result = filter_instance(cube)

        assert result.attributes.get("_create_args") == extra_args

    def test_extra_kwargs_passed(self):
        cube = sample_mesh_cube()
        extra_kwargs = {"key1": mock.sentinel.key1, "key2": mock.sentinel.key2}
        filter_instance = MyTestFilter(cube, **extra_kwargs)

        result = filter_instance(cube)

        assert result.attributes.get("_create_kwargs") == extra_kwargs

    def test_passed_cube_matches_input(self):
        cube = sample_mesh_cube()
        # Rename, to check the standard-name handling
        filter_instance = MyTestFilter(cube)

        result = filter_instance(cube)

        # Expect the result to match, *except* for the extra attributes added by the
        # test-filter operation.
        expected = cube.copy()
        expected.attributes.update(
            {"original-name": None, "_create_args": (), "_create_kwargs": {}}
        )

        assert result == expected


class TestFilterBadCallArgs:
    """
    Check that bad filter-call inputs are errored as expected.

    Effectively, this tests UnstructuredFilterABC._validate_call_argument, but it
    is adequate to test this integratively, which avoids implementation dependency.
    """

    @staticmethod
    def _sample_meshcube_and_filter():
        # Provide a standard meshcube and a filter built from it
        mesh = sample_mesh()
        # Name the mesh so that it has a predictable repr
        # -- i.e. so the repr does not include an object address.
        mesh.rename("example_mesh")
        cube = sample_mesh_cube(mesh=mesh)
        filter_instance = MyTestFilter(cube)
        return cube, filter_instance

    def test_noncube__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()

        with pytest.raises(Exception) as error:
            filter_instance(None)

        expected = (
            "TypeError('First argument must be a Cube, got : "
            "\"<class \\'NoneType\\'>\""
            ".')"
        )
        assert repr(error.value) == expected

    def test_nonmeshcube__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        nonmesh_cube = Cube([0])

        with pytest.raises(Exception) as error:
            filter_instance(nonmesh_cube)

        expected = (
            'ValueError(\'Input cube mesh "None" does not match '
            r"""the original filter mesh, "<Mesh: \'example_mesh\'>".')"""
        )
        assert repr(error.value) == expected

    def test_duplicate_mesh(self):
        """Check that filtering succeeds with a matching (but not same) mesh."""
        cube, filter_instance = self._sample_meshcube_and_filter()
        mesh = cube.mesh
        alternate_mesh = deepcopy(mesh)
        # Just ensure that the test meshes compare as expected
        # N.B. *** this is *not* the test ! ***
        assert alternate_mesh == mesh
        assert alternate_mesh is not mesh
        alternate_mesh_cube = sample_mesh_cube(mesh=alternate_mesh)

        # Check that the operation performs without error
        # N.B. *** this *is* the test ! ***
        assert filter_instance(alternate_mesh_cube) is not None

    def test_different_mesh__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        # make a mesh with a different number of faces
        other_mesh = sample_mesh()
        other_mesh.rename("other")
        other_mesh_cube = sample_mesh_cube(mesh=other_mesh)

        with pytest.raises(Exception) as error:
            filter_instance(other_mesh_cube)

        expected = (
            r"""ValueError('Input cube mesh "<Mesh: \'other\'>" does not match """
            r"""the original filter mesh, "<Mesh: \'example_mesh\'>".')"""
        )
        assert repr(error.value) == expected

    def test_different_location__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        # make meshcoords on a different location of the same mesh
        other_cube = sample_mesh_cube(mesh=cube.mesh, location="edge")

        with pytest.raises(Exception) as error:
            filter_instance(other_cube)

        expected = (
            r"""ValueError('Input cube location "edge" does not match the original """
            r"""filter location, "face".')"""
        )
        assert repr(error.value) == expected


class TestFilterBadInnerResults:
    """
    Check that bad results from an 'inner_filter' call are errored as expected.

    Effectively, this tests UnstructuredFilterABC._validate_inner_filter_result, but it
    is adequate to test this integratively, which avoids implementation dependency.
    """

    @staticmethod
    def _sample_meshcube_and_filter():
        # Use a mesh with equal numbers of faces + edges,
        # since otherwise we hit a shape mismatch when testing for a different location
        mesh = sample_mesh(n_faces=3, n_edges=3)
        # Name the mesh so that it has a predictable repr
        # -- i.e. so the repr does not include an object address.
        mesh.rename("example_mesh")
        cube = sample_mesh_cube(mesh=mesh)
        filter_instance = MyTestFilter(cube)
        return cube, filter_instance

    def test_noncube__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        filter_instance.fix_result_value_to(1)

        with pytest.raises(Exception) as error:
            filter_instance(cube)

        expected = (
            "TypeError('Inner-filter call returned a value which is not a Cube : 1.')"
        )
        assert repr(error.value) == expected

    def test_different_shape__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        filter_instance.fix_result_value_to(cube[1:])

        with pytest.raises(Exception) as error:
            filter_instance(cube)

        expected = (
            "ValueError('Inner-filter call returned a cube of a different shape to "
            "the input : (1, 3) != (2, 3).')"
        )
        assert repr(error.value) == expected

    def test_different_mesh__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        # Make the inner result a cube with an identical mesh, *except* for its name
        mesh2 = deepcopy(cube.mesh)
        mesh2.rename("other")
        cube_with_different_mesh = sample_mesh_cube(mesh=mesh2)
        filter_instance.fix_result_value_to(cube_with_different_mesh)

        with pytest.raises(Exception) as error:
            filter_instance(cube)

        expected = (
            'ValueError("Inner-filter call returned a cube of a different mesh to the '
            """input : <Mesh: 'other'> != <Mesh: 'example_mesh'>.")"""
        )
        assert repr(error.value) == expected

    def test_matching_mesh(self):
        """Check that an inner result with a different, but identical, mesh is OK."""
        cube, filter_instance = self._sample_meshcube_and_filter()
        # Make the inner return a cube with an identical mesh, *except* for its name
        mesh = cube.mesh
        mesh2 = deepcopy(mesh)
        # Just check that the 2 meshes behave as expected
        # N.B. *** this is *not* the test ! ***
        assert mesh2 == mesh
        assert mesh2 is not mesh
        # Fix the inner result value to a cube with the new mesh
        cube_with_equal_but_different_mesh = sample_mesh_cube(mesh=mesh2)
        filter_instance.fix_result_value_to(cube_with_equal_but_different_mesh)

        # Check that the operation runs successfully
        # N.B. *** this *is* the test ! ***
        assert filter_instance(cube) is not None

    def test_different_location__fails(self):
        cube, filter_instance = self._sample_meshcube_and_filter()
        # Make the inner return a cube on the same mesh, but a different location
        cube_with_different_location = sample_mesh_cube(mesh=cube.mesh, location="edge")
        filter_instance.fix_result_value_to(cube_with_different_location)

        with pytest.raises(Exception) as error:
            filter_instance(cube)

        expected = (
            """ValueError("Inner-filter call returned a cube of a different """
            """location to the input : 'edge' != 'face'.")"""
        )
        assert repr(error.value) == expected
