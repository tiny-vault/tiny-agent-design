"""Unified diff of ``TinyAgent`` from phase 1 to phase 2.

**What it shows.** The book grows one class phase by phase. A diff is the quickest way to see what a phase adds: lines starting with ``-`` are removed, lines with ``+`` are added.

**What changes in this diff.** ``__init__`` now takes an ``llm`` and creates a ``Trajectory``; ``run`` initialises the trajectory; and ``_step`` replaces the placeholder echo with a real model call (send the task as a user message, record the response, return its text). Memory, tools and planner are still ``None`` placeholders.

**How it works here.** ``tinyagents_diff`` renders HTML only in Jupyter, so the script prints ``difflib.unified_diff`` of the two classes' source (``old_cls`` against ``new_cls``). No model needed; the bare name ``tinyagents_diff`` on its own line mirrors the notebook cell.

Compare ``diff_agent_full_growth``, which diffs from phase 1 straight to phase 6.
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
