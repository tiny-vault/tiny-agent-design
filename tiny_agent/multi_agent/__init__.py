"""Multi-agent systems: agents that hand parts of a task to other agents.

One agent with many tools gets confused as the list grows; several small specialists each stay simple. In this
package a specialist (from ``tiny_agent.agents``) is wrapped in an ordinary function, and that function is
registered as a tool of another agent, the *orchestrator*. To the orchestrator a specialist is just a tool that takes
a question and returns text; behind it a whole ``TinyAgent`` runs its own loop with its own tools and memory.

* ``orchestrator.py``: ``create_orchestrator``, an agent whose only tools are ``ask_math_agent`` and
  ``ask_date_agent``.

* ``a2a.py``: ``A2AServer`` and ``A2AClient``, the A2A protocol (agent card and ``message/send`` over HTTP). With
  the ``a2a`` flag the orchestrator reaches its specialists through them instead of in-process.

Import from the files, for example ``from tiny_agent.multi_agent.orchestrator import create_orchestrator``.
"""
