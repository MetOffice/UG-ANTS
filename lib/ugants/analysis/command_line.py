# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Implementation for the fill missing points application."""

from iris.cube import CubeList

from ugants.abc import Application
from ugants.analysis.fill import fill_cube


class FillMissingPoints(Application):
    """Fill missing points for UGrid data.

    Uses the :class:`~ugants.analysis.fill.KDTreeFill` algorithm.
    """

    def __init__(self, source: CubeList, target_mask: CubeList = None):
        self.source = source

        if target_mask:
            if len(target_mask) != 1:
                raise ValueError(f"Expecting one target mask, found {len(target_mask)}")
            else:
                self.target_mask = target_mask[0]
        else:
            self.target_mask = None

    def run(self):
        """Fill missing points in the source cube.

        The :class:`ugants.analysis.fill.KDTreeFill` class is used with the provided
        target mask to fill missing points in the source cube. The filled result is
        stored in ``self.result``.
        """
        self.results = CubeList(
            fill_cube(cube, self.target_mask) for cube in self.source
        )
