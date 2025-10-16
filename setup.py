# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""
Part of the installation process for UG-ANTS.

Creates executable scripts that are callable from the command line.

"""
import os
from glob import glob

from setuptools import setup


def runnable_scripts(path):
    """Find everything in directory and put the names into an array."""
    files = glob(os.path.join(path, "*"))
    scripts = [fnme for fnme in files if os.path.isfile(fnme)]
    return scripts


if __name__ == "__main__":
    setup(
        scripts=runnable_scripts("bin"),
    )
