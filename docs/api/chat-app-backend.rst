Chat app backend
================

The FastAPI service in ``experimentation/chat-app/backend/`` that sits between the browser and
llama.cpp's router (see :doc:`../guides/chat-app`). ``server`` is the HTTP layer and its error
format; ``llama_client`` is the async client that lists, loads and chats with models. The
service is not part of ``code/tiny_agent`` and does not use it.

Run it with Docker (``node execute.mjs up``). To test it without Docker, use
``node execute.mjs test backend``, which runs it against a fake llama-server.

.. autosummary::
   :toctree: generated

   server
   llama_client
