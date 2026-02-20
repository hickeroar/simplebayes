#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = 'simplebayes'
copyright = '2026, Ryan Vennell'
author = 'Ryan Vennell'

try:
    import simplebayes
    version = getattr(simplebayes, '__version__', '2.1.0')
except ImportError:
    version = '2.1.0'
release = version

language = 'en'
exclude_patterns = ['_build']
add_function_parentheses = True
add_module_names = True
show_authors = False
pygments_style = 'sphinx'
todo_include_todos = True

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'simplebayesdoc'

latex_elements = {}
latex_documents = [
    (
        master_doc,
        'simplebayes.tex',
        'simplebayes Documentation',
        author,
        'manual',
    ),
]

man_pages = [
    (master_doc, 'simplebayes', 'simplebayes Documentation', [author], 1),
]

texinfo_documents = [
    (
        master_doc,
        'simplebayes',
        'simplebayes Documentation',
        author,
        'simplebayes',
        'One line description of project.',
        'Miscellaneous',
    ),
]

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']
