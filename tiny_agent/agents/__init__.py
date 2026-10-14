"""Ready-made agents: ``TinyAgent`` instances put together from the components for one job each.

``TinyAgent`` (``agent.py``) is the harness; ``components/`` holds its parts. An agent here is a factory that picks
a model, a tool registry, a memory and a planner for a narrow task, a *specialist*, and returns a new agent per call,
so no two uses share a memory. One file each:

* ``math_agent.py``: ``create_math_agent``, arithmetic with add, subtract and multiply as native tools, on
  ``gemma4-e4b`` with native thinking and ``NativeReAct``.
* ``date_agent.py``: ``create_date_agent``, the same setup with ``today`` and ``days_between`` as its tools.

Import from the files, for example ``from tiny_agent.agents.math_agent import create_math_agent``.
"""
