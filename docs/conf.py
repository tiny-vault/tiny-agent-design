"""Sphinx configuration for the TinyAgent documentation."""

import sys
from pathlib import Path

# Make the project modules importable for autodoc. Importing them is safe:
# tiny_agent/agent/tiny_agent.py has no side effects and tiny_agent/main.py guards its entry point.
# The repo root, for tiny_agent (with main, cli and llm_client inside it), experimentation.illustrated_agents and
# experimentation.trigger_ias. Those scripts
# guard their model calls, but importing them reads the llama_api_key secret: run
# `node execute.mjs up` once before building the docs.
sys.path.append(str(Path(__file__).resolve().parent.parent))
# The chat-app backend (server.py, llama_client.py) is not an installed package: put its folder on the
# path, appended so nothing it contains shadows another module. Importing server.py builds the FastAPI app
# but opens no connection; it needs fastapi and httpx (see docs/requirements.txt).
sys.path.append(str(Path(__file__).resolve().parent.parent / "experimentation" / "chat-app" / "backend"))

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

# Nitpicky mode (-n) warns about every type it cannot link. These come from the chat-app backend's
# pydantic and FastAPI types, whose manuals are not in intersphinx_mapping (adding them would make the
# docs build depend on two more websites); they are shown as plain text instead of links.
nitpick_ignore = [
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "fastapi.applications.FastAPI"),
    ("py:class", "annotated_types.MinLen"),
    ("py:class", "min_length=1"),
]

todo_include_todos = True

myst_enable_extensions = ["colon_fence"]
myst_fence_as_directive = ["mermaid"]

html_theme = "furo"
html_static_path = ["_static"]
