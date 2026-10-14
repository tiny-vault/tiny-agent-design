"""The agent's planners: the strategy that turns one model call into a think, act, observe loop.

With a planner, ``TinyAgent.run`` repeats step, act, observe up to ``max_steps`` times. A planner only
supplies the system-prompt section that teaches the loop (``prompt``) and the reader that pulls the
thought and the action out of a reply (``parse``). This package holds them, one file each:

* ``react.py``: ``ReAct``, which teaches a ``THOUGHT:`` / ``ACTION:`` text format and parses it (prompting tools).
* ``native_react.py``: ``NativeReAct``, an empty prompt and a pass-through parser, for native thinking and tool calls (learned tools).

Import from the files, for example ``from tiny_agent.components.planning.native_react import NativeReAct``;
both are also re-exported here, so ``from tiny_agent.components.planning import ReAct`` still works.
"""

from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.planning.react import ReAct

__all__ = ["NativeReAct", "ReAct"]
