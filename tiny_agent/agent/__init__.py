"""The TinyAgent harness package.

An LLM only turns text into text; the *harness* is the code around it that makes
it an agent. The pieces live here:

* :mod:`~tiny_agent.agent.tiny_agent`: the ``TinyAgent`` control loop and the
  slots for the LLM, memory, tools and planner.
* :mod:`~tiny_agent.agent.memory`: the ``Memory`` that stores the conversation
  sent to the model on every call.
* :mod:`~tiny_agent.agent.trajectory` and :mod:`~tiny_agent.agent.step`: the
  debugging record of what the agent did, one ``Step`` at a time.

The variants of ``Memory`` are not here: they are the ``tiny_agent.memory`` package,
next to this one (``TrimmingMemory``, ``SummarizationMemory``, ``RAGMemory``).

The LLM client (``LLM``, ``Response`` and the loaded-model check) is not here: it
is the separate, self-contained :mod:`tiny_agent.llm_client` package.

The demo entry point is ``main.py``, next to the package folders.
"""
