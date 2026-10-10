"""A small, self-contained client for chat models behind an OpenAI-compatible API.

Nothing here depends on the rest of the repository, so this folder can be copied
into another project and used on its own:

* :mod:`llm_client.llm`: the ``LLM`` class, which sends a conversation to a model.
* :mod:`llm_client.response`: the ``Response`` the model's reply is turned into.
* :mod:`llm_client.check`: asks the server which models are loaded
  (``python -m llm_client.check``).

Standard library only. See ``README.md`` in this folder.
"""
