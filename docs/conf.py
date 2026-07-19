"""Sphinx configuration for the bindcurve documentation."""

from importlib.metadata import PackageNotFoundError, version

project = "bindcurve"
author = "Jan Choutka"
copyright = "2026, Jan Choutka"

try:
    release = version("bindcurve")
except PackageNotFoundError:
    release = "0+unknown"

version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Authoring and notebook execution
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "tasklist",
]
myst_heading_anchors = 3
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 120

# Python API documentation
autosummary_generate = True
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
nitpicky = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# HTML output
html_theme = "pydata_sphinx_theme"
html_title = "bindcurve documentation"
html_show_sourcelink = False
html_context = {
    "github_user": "choutkaj",
    "github_repo": "bindcurve",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme_options = {
    "show_toc_level": 2,
    "navigation_depth": 4,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/choutkaj/bindcurve",
            "icon": "fa-brands fa-github",
        },
    ],
}
