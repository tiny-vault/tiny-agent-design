"""The agent's memory: the conversation history it sends to the model, and variants of it.

A model server is stateless, so the agent keeps the conversation itself and resends it on every call. This
package holds the ways of keeping it, one file each:

* ``memory.py``: ``Memory``, which keeps every message.
* ``trimming.py``: ``TrimmingMemory``, which keeps only the last two turns.

``Memory`` is also available straight from the package, so ``from tiny_agent.shared.memory import Memory``
works; ``TrimmingMemory`` is imported from its file (``tiny_agent.shared.memory.trimming``).
"""

from tiny_agent.shared.memory.memory import Memory

__all__ = ["Memory"]
