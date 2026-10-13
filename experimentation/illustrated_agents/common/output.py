"""Plain-text stand-ins for output that the book shows as Jupyter HTML."""

import difflib
from dataclasses import fields

from illustrated_agents.utils import _get_source


def show(response):
    # One Response field per line; the content gets its own block because it spans lines
    for f in fields(response):
        value = getattr(response, f.name)
        print(f"content:\n{value}" if f.name == "content" and value else f"{f.name}: {value}")


def print_diff(diff):
    # DiffViewer only renders HTML inside Jupyter, so print the same unified diff as plain text
    print("\n".join(difflib.unified_diff(
        _get_source(diff.old_cls).splitlines(),
        _get_source(diff.new_cls).splitlines(),
        diff.old_name,
        diff.new_name,
        lineterm="",
    )))
