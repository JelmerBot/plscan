# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
from setuptools_scm import get_version, Version

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = 'plscan'
copyright = '2025, Jelmer Bot'
author = 'Jelmer Bot'

# -- General configuration ---------------------------------------------------

_version = Version(get_version(Path(__file__).parent / '..'))
release = _version.public
version = _version.base_version
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
extensions = [
    "numpydoc",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_default_flags = ['members']
numpydoc_show_class_members = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
htmlhelp_basename = 'plscan_doc'
