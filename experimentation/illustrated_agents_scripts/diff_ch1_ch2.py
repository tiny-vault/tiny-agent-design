"""What changed in ``TinyAgent`` between chapter 1 and chapter 2 of the book, as a unified diff.

The book's ``illustrated_agents.chapters.ch2`` module ships ``tinyagents_diff``, a ``DiffViewer`` comparing ``ch1.TinyAgent`` with ``ch2.TinyAgent``. In Jupyter the bare name on the second line renders a side-by-side HTML diff. Outside Jupyter it does nothing visible, so when run as a script this module prints the same comparison as a plain-text unified diff (``difflib.unified_diff``) of the two classes' source code.

Reading the output:
    * The header lines ``--- ch1.TinyAgent`` and ``+++ ch2.TinyAgent`` name the old and new class.
    * ``@@ -a,b +c,d @@`` hunks give line ranges in each version.
    * Lines starting with ``-`` exist only in chapter 1; ``+`` only in chapter 2; a space, in both.

Names used (all from the ``illustrated_agents`` package):
    * ``tinyagents_diff``: the ``DiffViewer`` instance, with ``old_cls``, ``new_cls``, ``old_name`` and ``new_name``.
    * ``illustrated_agents.utils._get_source``: returns a class's source text. It is a private helper of the package, imported only inside the entry point.

Run:
    ``.venv/Scripts/python trigger_ias.py diff_ch1_ch2`` from the repo root. No model or Docker stack needed; it only reads source code. See also :mod:`~experimentation.illustrated_agents_scripts.diff_ch1_ch6`.
"""

from illustrated_agents.chapters.ch2 import tinyagents_diff
tinyagents_diff

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
