"""Configuration: build typed settings from the environment and a settings file.

The CLI, demo, model check and logger need the same defaults and parsing rules.
``settings.py`` defines ``Settings`` and ``Settings.from_env()`` so those
callers ask one object rather than interpret environment strings separately.
Real environment values win over the selected ``.env`` file, then defaults.
The old ``env.py`` and ``api_key_from_env`` names are not present here;
``Settings.resolve_api_key`` now reads ``TINYAGENT_API_KEY_FILE`` or
``TINYAGENT_API_KEY``.

Pass a ``Settings(...)`` object to configure a caller in code. Construction
still reads environment settings unless explicit field values override them;
this is not an environment-free object. This package initializer only
introduces the configuration module and makes no requests or file writes.
"""
