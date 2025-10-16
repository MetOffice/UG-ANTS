#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Save data in a format suitable for XIOS
***************************************

Using a UGrid input file, adds metadata and fixes data ordering to fit XIOS.

See Also
--------
ugants.io.xios_command_line
"""

from ugants.io.xios_command_line import ConvertToXIOS


def _parser():
    """Return the application's argument parser for use in the documentation build."""
    return ConvertToXIOS._parser()


if __name__ == "__main__":
    xios_app = ConvertToXIOS.from_command_line()
    xios_app.run()
    xios_app.save()
