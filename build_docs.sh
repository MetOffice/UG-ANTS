#!/bin/bash -l
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
#
# This is provided as a convenience - the main docs build is in the rose stem
# suite.

# Any changes to this file should also be made to the rose stem version.

set -eu
echo "Building UGANTS Documentation"
cd docs
make clean
# API docs
sphinx-apidoc ../lib/ "../lib/*tests" -o source/lib/ --force --module-first --no-toc
# CLI docs
sphinx-apidoc ../bin/ -o source/applications --force --module-first --templatedir source/_templates/ -d 2 --tocfile app_index
make html
