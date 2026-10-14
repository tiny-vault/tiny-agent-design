"""Configuration: the ``.env`` file and every setting in one place.

* ``env.py``: loads the ``.env`` file into the environment, and reads the API key (``api_key_from_env``) from
  ``TINYAGENT_API_KEY_FILE`` or ``TINYAGENT_API_KEY``.
* ``settings.py``: ``Settings``, every other ``TINYAGENT_*`` value with its default (``Settings.from_env()``), which the
  CLI, the demo, the model check and the logger use; pass a ``Settings(...)`` to configure one in code.
"""
