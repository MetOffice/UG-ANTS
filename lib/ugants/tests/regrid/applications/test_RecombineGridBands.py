# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from unittest import mock

import iris.coords
import iris.cube
import numpy as np

from ugants.regrid.applications import RecombineGridBands


class TestRun:
    def test_mixed_dtype(self):
        # Longitude will be common to all cubes:
        longitude = iris.coords.DimCoord(
            points=[-90, 90],
            bounds=np.array([[-180, 0], [0, 180]]),
            standard_name="longitude",
        )

        expected = iris.cube.Cube(np.ones((2, 4), dtype=np.float64))
        expected.add_dim_coord(longitude, 0)
        expected.add_dim_coord(
            iris.coords.DimCoord(
                [-60, -30, 30, 60],
                bounds=np.array([[-90, -45], [-45, 0], [0, 45], [45, 90]]),
                standard_name="latitude",
            ),
            1,
        )
        expected.attributes["history"] = "foo"

        # Create fake split cubes.  Note that first and second here have
        # different dtypes.
        first = iris.cube.Cube(np.ones((2, 2), dtype=np.float32))
        first.add_dim_coord(longitude, 0)
        first.add_dim_coord(
            iris.coords.DimCoord(
                [-60, -30],
                bounds=np.array([[-90, -45], [-45, 0]]),
                standard_name="latitude",
            ),
            1,
        )
        first.attributes["history"] = "foo"
        second = iris.cube.Cube(np.ones((2, 2), dtype=np.float64))
        second.add_dim_coord(longitude, 0)
        second.add_dim_coord(
            iris.coords.DimCoord(
                [30, 60], bounds=np.array([[0, 45], [45, 90]]), standard_name="latitude"
            ),
            1,
        )
        second.attributes["history"] = "foo"

        recombined_bands = RecombineGridBands(iris.cube.CubeList([first, second]))
        recombined_bands.run()
        actual = recombined_bands.results

        assert actual == expected

    def test_save_call(self):
        recombined_bands = RecombineGridBands("foo")
        recombined_bands.results = "bar"
        recombined_bands.output = "baz"
        with mock.patch(
            "ugants.regrid.applications.iris.fileformats.netcdf.save"
        ) as mock_save:
            recombined_bands.save()
        mock_save.assert_called_once_with("bar", "baz")
