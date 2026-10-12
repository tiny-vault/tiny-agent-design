"""Unified diff of ``TinyAgent`` from chapter 1 to chapter 6.

The book only diffs consecutive chapters, so ``tinyagents_diff`` is a ``DiffViewer`` built here and printed as plain text. No model needed.
"""

from illustrated_agents.chapters import ch1, ch6
from illustrated_agents.utils import DiffViewer

tinyagents_diff = DiffViewer(
    ch1.TinyAgent, ch6.TinyAgent,
    "Chapter 1 - TinyAgent", "Chapter 6 - TinyAgent"
)

# Runnable part: DiffViewer only renders HTML inside Jupyter, so print the
# same unified diff as plain text when this file is run as a script.
if __name__ == "__main__":
    import difflib

    from illustrated_agents.utils import _get_source

    d = tinyagents_diff
    print("\n".join(difflib.unified_diff(
        _get_source(d.old_cls).splitlines(),
        _get_source(d.new_cls).splitlines(),
        d.old_name,
        d.new_name,
        lineterm="",
    )))
