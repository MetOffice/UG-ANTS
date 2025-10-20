#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Horizontal regrid to UGrid mesh
*******************************

Regrids data from a source grid to a target UGrid mesh using
:class:`ugants.regrid.command_line.Regrid`.  The result is written to an output
file.  The application supports horizontal regridding only.

See Also
--------
ugants.regrid
"""

from ugants.regrid.command_line import Regrid


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return Regrid._parser()


if __name__ == "__main__":
    regrid_app = Regrid.from_command_line()
    regrid_app.run()
    regrid_app.save()
