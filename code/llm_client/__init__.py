"""A small, self-contained client for chat models behind an OpenAI-compatible API.

What is it? A language model runs on a server, and your program talks to it over
HTTP. Many servers (llama.cpp, Ollama, hosted APIs) accept the same request shape,
the "OpenAI-compatible" ``/v1/chat/completions`` dialect: you POST a list of
``messages`` and get back one reply. This package wraps that exchange so the rest
of the code can call ``llm.generate(messages)`` and never handle HTTP or JSON.

Why a separate package? Nothing here depends on the rest of the repository, so
this folder can be copied into another project and used on its own. Keeping the
model client apart from the agent also shows the split the book draws: the LLM
only maps text to text, and everything else is the harness around it.

The modules, in the order a request passes through them:

* ``llm_client.env``: loads settings (model name, server address) from a
  ``.env`` file into the environment.
* :mod:`llm_client.check`: asks the server which models are loaded
  (``python -m llm_client.check``), so a request can fail early with a clear message.
* :mod:`llm_client.llm`: the ``LLM`` class, which sends a conversation to a model.
* :mod:`llm_client.response`: the ``Response`` the model's reply is turned into.

Limitations: it is a blocking, non-streaming client with no retries, and it keeps
no conversation state. Standard library only. See ``README.md`` in this folder.
"""
