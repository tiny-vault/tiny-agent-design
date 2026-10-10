"""Sphinx configuration for the TinyAgent documentation."""

import sys
from pathlib import Path

# Make the project modules importable for autodoc. Importing them is safe:
# tiny_agent.py has no side effects and main.py guards its entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "TinyAgent"
copyright = "2026, TinyAgent contributors"
author = "TinyAgent contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# autodoc / autosummary
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# napoleon: Google-style docstrings only
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Keep section labels unique across documents.
autosectionlabel_prefix_document = True

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

todo_include_todos = True

myst_enable_extensions = ["colon_fence"]
myst_fence_as_directive = ["mermaid"]

html_theme = "furo"
html_static_path = ["_static"]
