# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Configuration file for UG-ANTS documentation build."""

import os
import sys
from datetime import datetime

import iris

sys.path.insert(0, os.path.abspath("../../bin"))  # For autodoc
sys.path.insert(0, os.path.abspath("../../lib"))  # For autodoc

import ugants

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "UG-ANTS"
project_copyright = f"2023 - {datetime.now().year}, Met Office"
author = "Model Inputs and Outputs team, Met Office"

version = ugants.__version__
release = ugants.__version__

iris_version = iris.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions taken from ANTS
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "iris": (f"https://scitools-iris.readthedocs.io/en/v{iris_version}", None),
}

extlinks = {
    "ugantsworkingpractices": (
        "https://code.metoffice.gov.uk/trac/ancil/wiki/ANTS/WorkingPractices/%s",
        "%s",
    ),
    "ticket": ("https://code.metoffice.gov.uk/trac/ancil/ticket/%s", "#%s"),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Don't check anchors on these pages, but check the page exists.
# For semver.org, we link to a particular paragraph, but the anchor
# for that paragraph is not in the HTML (and is instead presumably
# added by javascript).
linkcheck_anchors_ignore_for_url = ("https://semver.org/",)

# Set linkcheck to ignore certain URLs
# - Ignore UG-ANTS on GitHub while it is private
linkcheck_ignore = [r"https://github.com/MetOffice/UG-ANTS.*"]

# Napoleon config
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/MetOffice/UG-ANTS",
    "show_toc_level": 2,
}

# Make sphinx-copybutton skip all prompt characters in pygments highlighted
# code blocks.
copybutton_exclude = ".linenos, .gp, .go"
# Set default syntax highlighting to python:
highlight_language = "python"
