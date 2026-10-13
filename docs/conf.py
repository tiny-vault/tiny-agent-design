"""Sphinx configuration for the TinyAgent documentation."""

import sys
from pathlib import Path

# Make the project modules importable for autodoc. Importing them is safe:
# code/tiny_agent/shared/tiny_agent.py has no side effects and code/main.py guards its entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
# The repo root, for experimentation.illustrated_agents and trigger_ias. Appended, not
# inserted, so the stdlib `code` module is not shadowed by the repo's code/ folder. Those scripts
# guard their model calls, but importing them reads the llama_api_key secret: run
# `node execute.mjs up` once before building the docs.
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
# Show "Attributes:" sections as instance variables. Otherwise a dataclass such as
# Response is documented twice: once from its docstring, once from its fields.
napoleon_use_ivar = True

# Keep section labels unique across documents.
autosectionlabel_prefix_document = True

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

todo_include_todos = True

myst_enable_extensions = ["colon_fence"]
myst_fence_as_directive = ["mermaid"]

html_theme = "furo"
html_static_path = ["_static"]
