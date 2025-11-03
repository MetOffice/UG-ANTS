.. highlight:: shell

=================
How to Contribute
=================

Report Bugs
===========

Report bugs and request enhancement by emailing miao@metoffice.gov.uk and
ideally opening an issue on `UG-ANTS GitHub <https://github.com/MetOffice/UG-ANTS/issues/new>`_ as well.
If reporting a bug, add a recipe for repeating it, along with any error messages,
details of what version of UG-ANTS is being used etc. If requesting an enhancement
or behaviour change, describe the use case in detail.

Contribute Code
===============

All contributions to UG-ANTS are made via merges with the ``main``
branch of `UG-ANTS <https://github.com/MetOffice/UG-ANTS>`_, or with the ``trunk`` branch of `ancil/ug-contrib <https://code.metoffice.gov.uk/trac/ancil/browser/ug-contrib>`_.

New contributors to UG-ANTS (*either or both of the UG-ANTS and UG-Contrib repositories*) should complete the :ref:`Contributor Licence Agreement (CLA) <CLA>`, convert the completed form to PDF format, and send it to miao@metoffice.gov.uk.

UG-ANTS uses `pre-commit <https://pre-commit.com>`_ hooks.
If you are a first-time contributor, you may need to run the following command
once to install the relevant hooks::

    pre-commit run --all-files

You may need to activate an environment containing ``pre-commit`` before running.
