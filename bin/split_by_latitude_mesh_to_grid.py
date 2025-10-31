#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Split source and target by latitude application.
************************************************



"""

from ugants.regrid.applications import SplitMeshToGridByLatitude


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return SplitMeshToGridByLatitude._parser()


if __name__ == "__main__":
    regrid_app = SplitMeshToGridByLatitude.from_command_line()
    regrid_app.run()
    regrid_app.save()
