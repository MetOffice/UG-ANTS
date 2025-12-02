.. highlight:: console

======================
How to install UG-ANTS
======================

Initial Setup
=============

If you do not have Conda installed, you will need to follow these
`installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
to get started.  Here is the `getting started <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ guide on using conda.

Stable Installation
===================

A stable installation of UG-ANTS uses the same package versions for all
installations of a particular UG-ANTS version.  To create a stable UG-ANTS
environment, first use the `environment.lock` file from the working copy
of the UG-ANTS version you want to install to install the dependencies::

    $ conda create -p <path/to/install/to/ug_ants_x.y> --file environment.lock
    $ conda activate <ug_ants_x.y>

You can then install UG-ANTS into your environment using::

    $ python -m pip install .

Alternatively, if you want to make sure you are only using the versions of packages
specified in your environment, or are working on a platform without internet access,
you can use the ``--no-build-isolation`` option to pip as follows::

    $ python -m pip install --no-build-isolation .

Developer Installation
======================

For development work, you may want the most recent packages available, with only
key packages fixed to specific versions. In that case, use the ``environment.yml``
file to create the environment::

    $ conda env create -p <path/to/install/developer/environment> -f environment.yml
    $ conda activate <path/to/environment>
    $ python -m pip install --editable .

Testing the Installation
========================

To check UG-ANTS has been installed, activate the conda environment and run::

    $ ug-ants-version

This should tell you:
 - The path to your installation of UG-ANTS. This should point to inside your conda environment.
 - The version of UG-ANTS.
 - The path to the installation of iris. This should be in the same environment as UG-ANTS.
 - The version of iris.

To test the installation of UG-ANTS, cd into the installation directory (i.e. the first path
returned by the previous command without the ``__init__.py``).  Then run::

    $ cp -r <working/copy>/lib/ugants/tests/resources ./tests/.
    $ python -m pytest .

This will copy resources used by the unittests in UG-ANTS and then run the tests. All of the tests
should pass or be an expected failure. The resources directory is not used for anything other than
tests and can be removed after verifying the UG-ANTS installation.
