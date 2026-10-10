"""The TinyAgent harness package.

An LLM only turns text into text; the *harness* is the code around it that makes
it an agent. The pieces live here:

* :mod:`~tiny_agent.shared.tiny_agent`: the ``TinyAgent`` control loop and the
  slots for the LLM, memory, tools and planner.
* :mod:`~tiny_agent.shared.llm`: the ``LLM`` client that talks to the model.
* :mod:`~tiny_agent.shared.response`: the ``Response`` the model's reply is
  turned into.
* :mod:`~tiny_agent.shared.check`: asks the model server which models are loaded.

The demo entry point, ``python -m tiny_agent.main``, is one level up.
"""
