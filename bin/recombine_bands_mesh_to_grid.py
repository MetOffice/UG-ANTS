#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Recombine regridded bands into a single UGrid file
**************************************************



"""
from ugants.regrid.applications import RecombineGridBands


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return RecombineGridBands._parser()


if __name__ == "__main__":
    regrid_app = RecombineGridBands.from_command_line()
    regrid_app.run()
    regrid_app.save()
