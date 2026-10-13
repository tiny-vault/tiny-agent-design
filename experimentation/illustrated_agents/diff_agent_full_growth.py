"""Unified diff of ``TinyAgent`` from phase 1 to phase 6.

**What it shows.** The whole growth of the agent in one diff: lines starting with ``-`` are removed, lines with ``+`` are added.

**What changes in this diff.** The placeholders become real parts: ``__init__`` takes ``llm``, ``memory``, ``tools`` and ``planner`` and builds a system prompt from them. ``run`` becomes a loop of up to ``planner.max_steps`` steps (the autonomy loop). Each ``_step`` asks the model (thought), parses tool calls, returns the answer if the tools say it is done, and otherwise ``_execute_action`` runs the tool and stores the observation in memory.

**How it works here.** The book only diffs consecutive phases, so ``tinyagents_diff`` is a ``DiffViewer`` built here from ``ch1.TinyAgent`` and ``ch6.TinyAgent``. ``DiffViewer`` renders HTML only in Jupyter, so the script prints ``difflib.unified_diff`` of the two classes' source as plain text. No model needed.

Compare ``diff_agent_adds_llm``, which shows only the first step of this growth.
"""

from illustrated_agents.chapters import ch1, ch6
from illustrated_agents.utils import DiffViewer

tinyagents_diff = DiffViewer(
    ch1.TinyAgent, ch6.TinyAgent,
    "Phase 1 - TinyAgent", "Phase 6 - TinyAgent"
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
