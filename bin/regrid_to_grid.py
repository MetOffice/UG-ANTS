#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Horizontal regrid to regular grid application.
**********************************************

Regrids ugrid data to a target regular grid using
:class:`ugants.regrid.applications.MeshToGridRegrid`.  The result is written
to an output file.  The application supports horizontal regridding only.

See Also
--------
ugants.regrid

"""

from ugants.regrid.applications import MeshToGridRegrid


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return MeshToGridRegrid._parser()


if __name__ == "__main__":
    regrid_app = MeshToGridRegrid.from_command_line()
    regrid_app.run()
    regrid_app.save()
