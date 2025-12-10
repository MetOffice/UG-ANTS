#!/usr/bin/env python
"""Generate non-ugrid source data for unit testing."""

import iris
import ugants.tests.stock
import ugants.utils.cube

if __name__ == "__main__":
    cube = ugants.tests.stock.regular_grid_global_cube(144, 192)
    cube = ugants.utils.cube.prepare_for_save(cube)
    cube.attributes["source"] = (
        f"Created using ugants.tests.stock.regular_grid_global_cube at UG-ANTS v{ugants.__version__}"  # noqa: E501
    )
    iris.save(cube, "non_ugrid_data.nc")
