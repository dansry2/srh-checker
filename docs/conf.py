import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'srh-data-reports'
copyright = '2026, dansry2'
author = 'dansry2'
language = 'ru'

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'alabaster'
