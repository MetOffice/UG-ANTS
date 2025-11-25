# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import dask
import iris
import numpy as np
import pytest
from ugants.tests.stock import mesh_cube
from ugants.utils.cube import align_mask


def test_add_missing_mask():
    # Add mask of False values when none is present at all
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    assert not hasattr(acube.data, "mask")

    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == (n_faces,)
    assert (acube.data.mask == np.zeros((n_faces,), dtype=bool)).all()


def test_expand_single_false():
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    assert not hasattr(acube.data, "mask")

    acube.data = np.ma.masked_array(acube.data)
    assert hasattr(acube.data, "mask")
    assert not acube.data.mask

    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == (n_faces,)
    assert (acube.data.mask == np.zeros((n_faces,), dtype=bool)).all()


def test_no_change_falses():
    # Do nothing - mask already ok
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    assert not hasattr(acube.data, "mask")
    acube.data = np.ma.masked_array(
        acube.data, mask=np.zeros(acube.data.shape, dtype=bool)
    )
    assert hasattr(acube.data, "mask")
    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == (n_faces,)
    assert (acube.data.mask == np.zeros((n_faces,), dtype=bool)).all()


def test_no_change_trues():
    # Do nothing - mask already ok
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    assert not hasattr(acube.data, "mask")
    acube.data = np.ma.masked_array(
        acube.data, mask=np.ones(acube.data.shape, dtype=bool)
    )
    assert hasattr(acube.data, "mask")
    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == (n_faces,)
    assert (acube.data.mask == np.ones((n_faces,), dtype=bool)).all()


def test_single_element_check():
    n_faces = 1
    acube = mesh_cube(n_faces=n_faces)
    assert not hasattr(acube.data, "mask")

    acube.data = np.ma.masked_array(acube.data)
    assert hasattr(acube.data, "mask")
    assert not acube.data.mask

    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == (n_faces,)
    assert (acube.data.mask == np.zeros((n_faces,), dtype=bool)).all()


def test_align_mask_cubelist():
    # Do nothing - masks already ok
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    bcube = mesh_cube(n_faces=n_faces)
    cubes = iris.cube.CubeList([acube, bcube])
    for cube in cubes:
        assert not hasattr(cube.data, "mask")
        cube.data = np.ma.masked_array(
            cube.data, mask=np.ones(cube.data.shape, dtype=bool)
        )
        assert hasattr(cube.data, "mask")

    align_mask(cubes)

    for cube in cubes:
        assert hasattr(cube.data, "mask")
        assert cube.data.mask.shape == (n_faces,)
        assert (cube.data.mask == np.ones((n_faces,), dtype=bool)).all()


def test_add_missing_mask_cubelist():
    # Add masks to the cubes in a cube list when none are present at all
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    bcube = mesh_cube(n_faces=n_faces)
    cubes = iris.cube.CubeList([acube, bcube])
    for cube in cubes:
        assert not hasattr(cube.data, "mask")

    align_mask(cubes)

    for cube in cubes:
        assert hasattr(cube.data, "mask")
        assert cube.data.mask.shape == (n_faces,)
        assert (cube.data.mask == np.zeros((n_faces,), dtype=bool)).all()


def test_cube_numpy_data_boolean_False_adds_mask():
    masked_array = np.ma.masked_array([1, 2, 3], mask=False)
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    acube.data = masked_array

    assert not acube.has_lazy_data()
    align_mask(acube)
    assert not acube.has_lazy_data()

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == acube.data.shape
    assert (acube.data.mask == np.zeros(acube.data.shape, dtype=bool)).all()


def test_cube_dask_data_boolean_False_adds_mask_unrealised():
    masked_array = dask.array.ma.masked_array([1, 2, 3], mask=False)
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    acube.data = masked_array

    assert acube.has_lazy_data()
    align_mask(acube)
    assert acube.has_lazy_data()

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == acube.data.shape
    assert (acube.data.mask == np.zeros(acube.data.shape, dtype=bool)).all()


def test_align_mask_for_numpy_arrays_keeps_type():
    masked_array = np.ma.masked_array([1, 2, 3], mask=False)
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    acube.data = masked_array

    pre_fix_type = acube.core_data()
    assert isinstance(pre_fix_type, np.ndarray)
    assert not acube.has_lazy_data()

    align_mask(acube)

    post_fix_type = acube.core_data()
    assert isinstance(post_fix_type, np.ndarray)
    assert not acube.has_lazy_data()


def test_align_mask_for_dask_arrays_keeps_type():
    masked_array = dask.array.ma.masked_array([1, 2, 3], mask=False)
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    acube.data = masked_array

    pre_fix_type = acube.core_data()
    assert isinstance(pre_fix_type, dask.array.core.Array)
    assert acube.has_lazy_data()

    align_mask(acube)

    assert acube.has_lazy_data()
    post_fix_type = acube.core_data()
    assert isinstance(post_fix_type, dask.array.core.Array)


def test_cube_numpy_data_mask_unmatched():
    error_string = r"Mask and data not compatible: data size is 3, mask size is 2."
    with pytest.raises(np.ma.core.MaskError, match=error_string):
        np.ma.masked_array([1, 2, 3], mask=[0, 1])


def test_cube_dask_data_mask_unmatched():
    error_string = \
        r"Mask and data not compatible: data shape is \(3,\), and mask shape is \(2,\)."
    with pytest.raises(np.ma.core.MaskError, match=error_string):
        dask.array.ma.masked_array([1, 2, 3], mask=[0, 1])


def test_cube_numpy_data_mask_single_true():
    masked_array = np.ma.masked_array([1, 2, 3], mask=True)
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    acube.data = masked_array

    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == acube.data.shape
    assert np.ma.all(np.ma.getmask(acube.core_data()))


def test_cube_dask_data_mask_single_true():
    masked_array = dask.array.ma.masked_array([1, 2, 3], mask=True)
    n_faces = 3
    acube = mesh_cube(n_faces=n_faces)
    acube.data = masked_array

    align_mask(acube)

    assert hasattr(acube.data, "mask")
    assert acube.data.mask.shape == acube.data.shape
    assert dask.array.all(dask.array.ma.getmaskarray(acube.core_data()))

