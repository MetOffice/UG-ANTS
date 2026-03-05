.. meta::
   :description lang=en: Tutorial on adding sources for testing
   :keywords: sources, rose stem, development, tutorial
   :property=og:locale: en_GB

.. highlight:: console

.. _managing-sources:

Managing rose stem sources
==========================

Initial setup
-------------

Before running the rose stem suite, the environment variable
``UG_ANTS_SOURCES_DIRECTORY_DEFAULT`` needs to be set to point to a suitable
central location for the :ref:`sources directory <sources-directory-structure>`
corresponding to the version of UG-ANTS being installed. Source data to be used
in rose stem testing must be added to this directory. A version specific module
file, site-specific ``ug-ants-launch`` script, or any other appropriate method
can be used for setting the environment variable.

This process should be repeated for ug-ancillary-file-science, with the
``UG_CONTRIB_SOURCES_DIRECTORY_DEFAULT`` environment variable.

.. _sources-directory-structure:

Sources directory structure
---------------------------

It is recommended to keep a complete set of sources for the current release.  It
may also be necessary to store a set of sources for head of trunk, and a number
of previous releases.  The following directory structure is suggested for
UG-ANTS core and ug-ancillary-file-science::

  UG-ANTS
    ├── developer
    │   ├── contrib
    │   │   └── <full_source_files>
    │   └── core
    │       └── <full_source_files>
    └── release
        └── X.Y.Z
            ├── contrib
            │   └── <full_source_files>
            └── core
                └── <full_source_files>

Development changes
-------------------

If a contributor has a change that adds or changes sources, then they
should:

1. Add new source files changes to a local directory.  This only needs to
   be the source files needed for any rose stem tests affected by the change,
   rather than the full set of sources.
2. Point to the new source files directly from the relevant ``rose-app.conf``.
3. Re-run the rose stem suite to generate new KGOs, see :ref:`managing-KGOs`.
4. Seek science owner approval for source and KGO changes.
5. When the ticket is complete, please include a summary of the source and KGO
   changes on the ticket template.
