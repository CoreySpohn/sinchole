"""Sphinx configuration file."""

from importlib.metadata import version as get_version

project = "sinchole"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"
release = get_version("sinchole")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "IPython.sphinxext.ipython_console_highlighting",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

templates_path = []
exclude_patterns = ["_build", ".build", "Thumbs.db", ".DS_Store"]

language = "en"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autodoc_typehints = "description"
napoleon_use_ivar = True

# Silence the harmless generated _version import note.
suppress_warnings = ["autoapi.python_import_resolution"]

myst_enable_extensions = ["amsmath", "dollarmath"]
myst_fence_as_directive = ["mermaid"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_title = "sinchole"

html_theme_options = {
    "repository_url": "https://www.github.com/CoreySpohn/sinchole",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "auto"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_output_stderr = "remove"
