from datetime import date

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src/material_topopt"))

#os.environ["MPLBACKEND"] = "Agg"  # avoid tkinter import errors on rtfd.io

# -- Project information -----------------------------------------------------

project = "material_topopt"

extensions = ['sphinx.ext.todo','sphinx.ext.napoleon']
#extensions = [
#    "sphinx.ext.autodoc",
#    "sphinx.ext.autosummary",
#    "sphinx.ext.doctest",
#    "sphinx.ext.todo",
#    "numpydoc",
#    "sphinx.ext.ifconfig",
#    "sphinx.ext.viewcode",
#    "sphinx.ext.imgmath",
#]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The root toctree document
# master_doc = "index"  # NOTE: will be changed to `root_doc` in sphinx 4

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
#pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "classic"
html_theme_options = {'body_max_width': '99%'}
