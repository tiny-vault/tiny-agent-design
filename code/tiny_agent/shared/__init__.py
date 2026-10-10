"""The TinyAgent harness package.

An LLM only turns text into text; the *harness* is the code around it that makes
it an agent. The pieces live here:

* :mod:`~tiny_agent.shared.tiny_agent`: the ``TinyAgent`` control loop and the
  slots for the LLM, memory, tools and planner.

The LLM client (``LLM``, ``Response`` and the loaded-model check) is not here: it
is the separate, self-contained :mod:`llm_client` package at the repository root.

The demo entry point is ``code/main.py``, next to the package folders.
"""
