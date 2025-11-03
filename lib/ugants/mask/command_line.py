# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Implementation for the mask generation application."""

from dataclasses import dataclass
from typing import Literal

import iris.cube
import numpy as np

from ugants.abc import Application


@dataclass
class GenerateMask(Application):
    """Generate a land mask or sea mask from a land area fraction input."""

    land_fraction: iris.cube.CubeList
    mask_type: Literal["land", "sea"]

    def run(self):
        """Run the generate mask app.

        * Extract the ``land_area_fraction`` data from the provided
          :class:`~iris.cube.CubeList`
        * Apply the :func:`~ugants.mask.command_line.derive_mask` function to
          the extracted data
        """
        land_fraction_cube = self.land_fraction.extract_cube("land_area_fraction")
        self.results = derive_mask(land_fraction_cube, self.mask_type)


def derive_mask(land_fraction, mask_type):
    """Create the specified binary mask.

    Parameters
    ----------
    land_fraction : iris.cube.Cube
        A :class:`~iris.cube.Cube` containing ``land_area_fraction`` data from
        which to derive the binary mask.
    mask_type : str
        The type of binary mask to produce: ``"land"`` or ``"sea"``.

    Returns
    -------
    iris.cube.Cube
        A new :class:`~iris.cube.Cube` with the derived binary mask.
    """
    mask = land_fraction.copy(data=np.zeros_like(land_fraction.data, dtype="int8"))

    if mask_type == "land":
        mask.data[land_fraction.data > 0] = 1
        mask.rename("land_binary_mask")
    elif mask_type == "sea":
        mask.data[land_fraction.data < 1] = 1
        mask.rename("sea_binary_mask")
    else:
        raise ValueError(
            f"Provided mask_type '{mask_type}' is not valid. Acceptable "
            "values are either 'land' or 'sea'"
        )

    mask.attributes["valid_min"] = 0
    mask.attributes["valid_max"] = 1

    return mask
