"""Variants of the agent's memory: other ways of keeping the conversation it sends to the model.

A model server is stateless, so the agent keeps the conversation itself and resends it on every call. The harness
only knows ``BaseMemory`` (``tiny_agent/components/memory/memory.py``): the message list and ``get_messages``. This package
holds the memories built on it, one file each:

* ``memory.py``: ``Memory``, the plain one, which keeps every message.
* ``trimming.py``: ``TrimmingMemory``, which keeps only the last two turns.
* ``summarization.py``: ``SummarizationMemory``, which keeps a running summary written by the model.
* ``rag.py``: ``RAGMemory``, which puts the documents most similar to each question in front of it.
* ``memory_bank.py``: ``MemoryBank``, which keeps event summaries and a user portrait that fade unless recalled.

Import ``Memory`` from its file, ``from tiny_agent.components.memory.memory import Memory``; it is also re-exported here, so
``from tiny_agent.components.memory import Memory`` still works. ``TrimmingMemory``, ``SummarizationMemory`` and
``RAGMemory`` are imported from their files (``tiny_agent.components.memory.trimming``, ``.summarization``, ``.rag``, ``.memory_bank``).
"""

from tiny_agent.components.memory.memory import Memory

__all__ = ["Memory"]
