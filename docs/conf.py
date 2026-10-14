"""Sphinx configuration for the TinyAgent documentation."""

import sys
from pathlib import Path

# Make the project modules importable for autodoc. Importing them is safe:
# tiny_agent/agent.py has no side effects and tiny_agent/main.py guards its entry point.
# The repo root, for tiny_agent (with main, cli, memory, tools, client and config inside it), experimentation.illustrated_agents and
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

# Pydantic's and pydantic-settings' own types appear on the Settings page; their manuals are not in
# intersphinx_mapping (same reason as above), so those names are shown as plain text.
nitpick_ignore_regex = [
    ("py:class", r"pydantic.*"),
    ("py:exc", r"pydantic.*"),
    ("py:class", r"(EnvPrefixTarget|DotenvType|CliSettingsSource|PathType|PydanticBaseSettingsSource)"),
    # napoleon reads the first line of a property docstring in tiny_agent as its type; the text is fine
    ("py:class", r"(Nothing|Which memory the demo uses.*|Which tools the demo binds)"),
]

# Look: a GitBook-style book. Furo gives the left sidebar (the toctree captions become its section
# headings), the search box, the 'On this page' column and the previous/next links; _static/gitbook.css
# restyles them (colours, type, rounded cards and hints).
html_title = "TinyAgent"
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["gitbook.css"]
html_show_sphinx = False
html_show_copyright = False
html_theme_options = {
    "source_repository": "https://github.com/tiny-vault/tiny-agent-design/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["edit"],
    "light_css_variables": {
        "color-brand-primary": "#3b6fd4",
        "color-brand-content": "#3b6fd4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7aa2f7",
        "color-brand-content": "#7aa2f7",
    },
}
