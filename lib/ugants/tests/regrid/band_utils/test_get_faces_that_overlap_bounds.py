# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from ugants.regrid.band_utils import (
    get_faces_that_overlap_bounds,
    mesh_to_cube,
)
from ugants.tests.stock import regular_lat_lon_mesh


class TestFacesThatOverlapBounds:
    # The regular_lat_lon_mesh face ordering is as follows.
    #
    # |2|5|8|
    # |1|4|7|
    # |0|3|6|
    #
    mesh = regular_lat_lon_mesh(0, 10, 0, 15, (3, 3))
    cube = mesh_to_cube(mesh)

    def test_bottom_boundary_nodes(self):
        # For these bounds we would expect faces [0,3,6]
        bounds = (0, 1)
        expected = [True, False, False, True, False, False, True, False, False]
        result = get_faces_that_overlap_bounds(self.cube, bounds, index=0)
        assert all(expected == result)

    def test_shared_nodes(self):
        # For these bounds we would expect faces [1,2,4,5,7,8]
        bounds = (9, 11)
        expected = [False, True, True, False, True, True, False, True, True]
        result = get_faces_that_overlap_bounds(self.cube, bounds, index=0)
        assert all(expected == result)

    def test_top_boundary_nodes(self):
        # For these bounds we would expect faces [2,5,8]
        bounds = (14, 15)
        expected = [False, False, True, False, False, True, False, False, True]
        result = get_faces_that_overlap_bounds(self.cube, bounds, index=0)
        assert all(expected == result)
