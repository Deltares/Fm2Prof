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
#from jupyter_sphinx_theme import *
#init_theme()
import json

# -- Project information -----------------------------------------------------

project = 'FM2PROF'
copyright = '2020, Deltares'
author = 'Koen Berends'
contact = 'koen.berends@deltares.nl'

# The full version, including alpha/beta/rc tags
import fm2prof
release = fm2prof.Project().__version__

# To enable to inject project name in source 
rst_epilog = f"""
.. |project| replace:: {project} 
.. |release| replace:: {release}
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_materialdesign_theme'

html_theme_options = {
    # Specify a list of menu in Header.
    # Tuples forms:
    #  ('Name', 'external url or path of pages in the document', boolean, 'icon name')
    #
    # Third argument:
    # True indicates an external link.
    # False indicates path of pages in the document.
    #
    # Fourth argument:
    # Specify the icon name.
    # For details see link.
    # https://material.io/icons/
    'header_links' : [
        ('Home', 'index', False, 'home'),
        ("Deltares", "https://deltares.nl", True, 'launch'),
    ],

    # Customize css colors.
    # For details see link.
    # https://getmdl.io/customize/index.html
    #
    # Values: amber, blue, brown, cyan deep_orange, deep_purple, green, grey, indigo, light_blue,
    #         light_green, lime, orange, pink, purple, red, teal, yellow(Default: indigo)
    'primary_color': 'blue',
    # Values: Same as primary_color. (Default: pink)
    'accent_color': 'amber',

    # Customize layout.
    # For details see link.
    # https://getmdl.io/components/index.html#layout-section
    'fixed_drawer': True,
    'fixed_header': True,
    'header_waterfall': True,
    'header_scroll': False,

    # Render title in header.
    # Values: True, False (Default: False)
    'show_header_title': False,
    # Render title in drawer.
    # Values: True, False (Default: True)
    'show_drawer_title': True,
    # Render footer.
    # Values: True, False (Default: True)
    'show_footer': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for LaTeX output -------------------------------------------------

latex_docclass = {
   'howto': 'article',
   'manual': 'deltares_report',
}

latex_additional_files = [
'latex_additional_files/sphinx.sty']


latex_elements = {
'papersize': 'a4paper',
'tableofcontents': '' ,
'fncychap': '',
'geometry': '',
'maketitle': '\\deltarestitle',
'fontpkg': '',
'preamble':r"""
\fancypagestyle{normal}{\pagestyle{plain}}
\partner{}
\client{}
\contact{koen.berends@deltares.nl}
\reference{}
\keywords{FM2D, FM1D, DHydro}
\projectnumber{..}
\documentid{}
\status{Automatically generated from source}
\disclaimer{}
\authori{Koen Berends}
\revieweri{}
\approvali{}
\publisheri{}
\organisationi{Deltares}
"""+
rf"""
\subtitle{{Manual for version {release}}}
\version{{{release}}}
\versioni{{{release}}}
"""
}

def setup(app):
    app.add_css_file('custom.css')

def generate_files_chapters():
    with open('../../fm2prof/configurationfile_template.json', 'r') as f:
        data = json.load(f)

        with open('chapters/files.rst', 'w') as f:
            f.write(f"""Files
========

Configuration file
-------------------
Default settings

.. code-block:: text

""")
            for line in fm2prof.Project().get_inifile()._print_configuration(data).splitlines():
                f.write(f"\t{line}\n")

generate_files_chapters()