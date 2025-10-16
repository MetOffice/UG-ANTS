#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Extract a single mesh from a UGrid netCDF file
==============================================

Load, extract and save a single mesh from a UGrid netCDF
file containing multiple meshes.

See Also
--------
ugants.io.applications.ExtractSingleMesh
"""
from ugants.io.applications import ExtractSingleMesh


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return ExtractSingleMesh._parser()


if __name__ == "__main__":
    extract_mesh_app = ExtractSingleMesh.from_command_line()
    extract_mesh_app.save()
