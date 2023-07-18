# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys

sys.path.append("..")


project = "FGVC"
copyright = "2023, Rail Chamidullin, Lukas Picek"
author = "Rail Chamidullin, Lukas Picek"
release = "1.4.0-dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # for pulling documentation from docstrings in a semi-automatic way
    # 'sphinx.ext.autosummary',  # for generating function/method/attribute summary lists
    "myst_parser",  # for .md support, see https://github.com/executablebooks/MyST-Parser
    "sphinx.ext.napoleon",  # for NumPy and Google style docstrings support
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# https://github.com/readthedocs/sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"  # alabaster
html_static_path = ["_static"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
