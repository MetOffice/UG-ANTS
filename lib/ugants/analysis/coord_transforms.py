# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Provides functions to convert between coordinate reference systems."""
import warnings

import numpy as np
from iris.cube import Cube


def convert_to_cartesian(source: Cube) -> np.ndarray:
    """Convert the data locations of an iris UGrid cube to a 3D Cartesian point cloud.

    The coordinate system of the source cube is assumed to be spherical,
    and the points are mapped to the unit sphere.

    Parameters
    ----------
    source : :class:`iris.cube.Cube`
        A UGrid cube to transform

    Returns
    -------
    :class:`numpy.ndarray`
        An nx3 array representing the coordinates of the n data locations in 3D space.

    Warning
    -------
    This function assumes a spherical geocentric coordinate system for conversion to
    3D cartesian coordinates. If the provided cube is not defined on this
    coordinate system then unexpected results may occur.
    """
    # Raise warning at stacklevel 2, i.e. the warning will refer to the function
    # that called convert_to_cartesian
    warnings.warn(
        "Assuming a spherical geocentric coordinate system for conversion to "
        "3D cartesian coordinates. If the provided cube is not defined on this "
        "coordinate system then unexpected results may occur.",
        stacklevel=2,
    )
    lat = source.coord(axis="y").points * np.pi / 180.0
    lon = source.coord(axis="x").points * np.pi / 180.0
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array((x, y, z)).T
