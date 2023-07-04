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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'DrugEx'
copyright = '2022, Xuhan Liu, Sohvi Lukkonen, Helle van den Maagdenberg, Linde Schoenmaker, Martin Šícho, Olivier Béquignon'
author = 'Xuhan Liu, Sohvi Lukkonen, Helle van den Maagdenberg, Linde Schoenmaker, Martin Šícho, Olivier Béquignon'

# The full version, including alpha/beta/rc tags
import importlib.util
spec = importlib.util.spec_from_file_location("drugex.about", "../drugex/about.py")
about = importlib.util.module_from_spec(spec)
spec.loader.exec_module(about)
release = about.VERSION
version = f'v{release}'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

intersphinx_mapping = {'python': ('https://docs.python.org/3.9', None)}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False
}

# napoleon settings
# https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Number the code samples, tables, figures, etc.
numfig = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'], }

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If false, no module index is generated.
html_domain_indices = True

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'any'