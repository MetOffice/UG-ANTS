#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Horizontal regrid UGrid mesh to UGrid mesh application.
*******************************************************

Regrids data from a source UGrid mesh to a target UGrid mesh using
:class:`ugants.regrid.command_line.RegridMeshToMesh`.  The result is written
to an output file. The application supports horizontal regridding only.  The
default regrid algorithm used is area weighted (conservative).

See Also
--------
ugants.regrid

"""

from ugants.regrid.command_line import RegridMeshToMesh


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return RegridMeshToMesh._parser()


if __name__ == "__main__":
    RegridMeshToMesh_app = RegridMeshToMesh.from_command_line()
    RegridMeshToMesh_app.run()
    RegridMeshToMesh_app.save()
