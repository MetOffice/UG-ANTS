#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Generate land and sea binary masks
**********************************

Using a UGrid land_area_fraction input file, generate either a land_binary_mask,
or a sea_binary_mask. Masks follow the cf convention where:

- land_binary_mask: 1 = land, 0 = sea
- sea_binary_mask: 1 = sea, 0 = land

See Also
--------
ugants.mask
"""

from ugants.mask.command_line import GenerateMask


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return GenerateMask._parser()


if __name__ == "__main__":
    generate_mask_app = GenerateMask.from_command_line()
    generate_mask_app.run()
    generate_mask_app.save()
