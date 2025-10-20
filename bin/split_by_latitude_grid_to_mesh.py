#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Split source and target by latitude application.
================================================

Split source data and target mesh into latitude bands, to facilitate regridding from
regular grid to unstructured mesh. Source bands are generated to fully enclose the
bands of the target mesh. A mesh mapping dataset is also created, which maps each cell
in the target mesh to its corresponding latitude band.

See Also
--------
ugants.regrid.command_line.SplitGridToMeshByLatitude
"""

from ugants.regrid.command_line import SplitGridToMeshByLatitude


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return SplitGridToMeshByLatitude._parser()


if __name__ == "__main__":
    regrid_app = SplitGridToMeshByLatitude.from_command_line()
    regrid_app.run()
    regrid_app.save()
