#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Recombine regridded bands into a single UGrid file
==================================================

Recombine regridded latitude bands into a UGrid dataset covering the full domain of the
target mesh from which the bands were generated. A mesh mapping dataset is required to
map the individual bands onto the correct locations on the target mesh, based on the
``band_number`` attribute on each latitude band.

See Also
--------
ugants.regrid.command_line.RecombineMeshBands
"""

from ugants.regrid.command_line import RecombineMeshBands


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return RecombineMeshBands._parser()


if __name__ == "__main__":
    regrid_app = RecombineMeshBands.from_command_line()
    regrid_app.run()
    regrid_app.save()
