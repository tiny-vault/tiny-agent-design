"""The model clients: ``llm.py`` is ``LLM`` (chat) and ``embedding.py`` is ``EmbeddingModel`` (vectors), both sending their request with ``post_json`` in ``http.py``.

``check.py`` asks the server which models are loaded; typed ``.env`` settings are in ``tiny_agent/config/settings.py`` and the ``Response`` type in ``tiny_agent/components/client/response.py``.

The client boundary converts model-server JSON into the objects the harness
uses. Chat and embedding clients hold connection settings but do not own
conversation memory. Import their classes from their modules; this initializer
exports no client objects and makes no HTTP requests.
"""
