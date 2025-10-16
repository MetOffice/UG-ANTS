#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Fill missing points
*******************

Fills missing points on a UGrid NetCDF file using the KDtree algorithm.
Also supports user provided target masks.

See Also
--------
ugants.analysis.command_line.FillMissingPoints
ugants.analysis.fill
"""

from ugants.analysis.command_line import FillMissingPoints


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return FillMissingPoints._parser()


if __name__ == "__main__":
    fill_application = FillMissingPoints.from_command_line()
    fill_application.run()
    fill_application.save()
