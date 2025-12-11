#!/usr/bin/env python
"""Generate non-ugrid source data for unit testing."""

import iris
import iris.coord_systems
import ugants.tests.stock
import ugants.utils.cube


def main():
    """Create a cube containing synthetic data on an n96 grid."""
    cube = ugants.tests.stock.regular_grid_global_cube(144, 192)
    cube = ugants.utils.cube.prepare_for_save(cube)
    coord_system = iris.coord_systems.GeogCS(6371229.0)
    cube.coord(axis="x").coord_system = coord_system
    cube.coord(axis="y").coord_system = coord_system
    cube.attributes["source"] = (
        f"Created using ugants.tests.stock.regular_grid_global_cube at UG-ANTS v{ugants.__version__}"  # noqa: E501
    )
    return cube


if __name__ == "__main__":
    cube = main()
    iris.save(cube, "non_ugrid_data.nc")
