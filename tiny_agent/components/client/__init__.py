"""The model clients: ``llm.py`` is ``LLM`` (chat) and ``embedding.py`` is ``EmbeddingModel`` (vectors), both sending their request with ``post_json`` in ``http.py``.

``check.py`` asks the server which models are loaded; the ``.env`` loader is in ``tiny_agent/config`` and the ``Response`` type in ``tiny_agent/components/client/response.py``.
"""
