# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Implementation for the fill missing points application."""

from dataclasses import dataclass

import iris.cube

from ugants.abc import Application
from ugants.analysis.fill import KDTreeFill


@dataclass
class FillMissingPoints(Application):
    """Fill missing points for UGrid data.

    Uses the :class:`~ugants.analysis.fill.KDTreeFill` algorithm.
    """

    source: iris.cube.CubeList
    target_mask: iris.cube.CubeList = None

    def run(self):
        """Fill missing points in the source cube.

        The :class:`ugants.analysis.fill.KDTreeFill` class is used with the provided
        target mask to fill missing points in the source cube. The filled result is
        stored in ``self.result``.
        """
        if len(self.source) != 1:
            raise ValueError(f"Expecting one cube, found {len(self.source)}")
        else:
            source_cube = self.source[0]

        if self.target_mask:
            if len(self.target_mask) != 1:
                raise ValueError(f"Expecting one cube, found {len(self.target_mask)}")
            else:
                target_mask_cube = self.target_mask[0]
        else:
            target_mask_cube = None

        filler = KDTreeFill(source_cube, target_mask_cube)
        self.results = filler(source_cube)
