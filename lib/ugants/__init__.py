# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

"""UG-ANTS: A library for generating ancillary files targeting unstructured grids."""

import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("UG-ANTS")
except PackageNotFoundError as exc:
    warnings.warn(
        f"{exc}. Unknown UG-ANTS version. "
        "Consider installing UG-ANTS into your environment.",
        stacklevel=0,
    )
    __version__ = "unknown"
