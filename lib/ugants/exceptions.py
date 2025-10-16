# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Exceptions specific to UG-ANTS."""

PROVISIONAL_WARNING_MESSAGE = (
    "The results output by UG-ANTS are provisional and subject to change."
)


class ProvisionalWarning(UserWarning):
    """A warning to inform users that results are provisional."""

    pass
