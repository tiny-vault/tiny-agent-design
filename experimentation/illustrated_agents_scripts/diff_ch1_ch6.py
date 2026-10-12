"""How ``TinyAgent`` grew from chapter 1 to chapter 6 of the book, as a unified diff.

The book ships a ready-made diff only between consecutive chapters. This module builds its own ``DiffViewer`` spanning chapters 1 to 6, so you can see everything the agent gained across those chapters in one comparison. In Jupyter, :data:`tinyagents_diff` renders as side-by-side HTML; when run as a script this module prints the same comparison as a plain-text unified diff (``difflib.unified_diff``) of the two classes' source code.

Reading the output:
    * The header lines ``--- Chapter 1 - TinyAgent`` and ``+++ Chapter 6 - TinyAgent`` name the old and new class.
    * ``@@ -a,b +c,d @@`` hunks give line ranges in each version.
    * Lines starting with ``-`` exist only in chapter 1; ``+`` only in chapter 6; a space, in both.

Run:
    ``.venv/Scripts/python trigger_ias.py diff_ch1_ch6`` from the repo root. No model or Docker stack needed; it only reads source code. See also :mod:`~experimentation.illustrated_agents_scripts.diff_ch1_ch2`.
"""

from illustrated_agents.chapters import ch1, ch6
from illustrated_agents.utils import DiffViewer

tinyagents_diff = DiffViewer(
    ch1.TinyAgent, ch6.TinyAgent,
    "Chapter 1 - TinyAgent", "Chapter 6 - TinyAgent"
)
"""``DiffViewer`` comparing ``ch1.TinyAgent`` (old) with ``ch6.TinyAgent`` (new).

Its attributes ``old_cls``, ``new_cls``, ``old_name`` and ``new_name`` feed the plain-text diff printed by the entry point; the names become the diff's ``---`` and ``+++`` headers. Building it only stores the classes and names; no diff is computed until it is displayed or printed.

:meta hide-value:
"""

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
