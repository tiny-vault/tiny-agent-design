"""The functions the agent can call as tools, one per file: ``multiply.py``, ``divide.py``.

Add a new tool as a new file here, then register it with ``add_tool`` (``default.py`` does it for the built-in ones).
These functions accept text because this project advertises string parameters
in its tool schemas; native APIs can support other parameter types. ``float``
conversion happens inside each arithmetic function. Non-numeric arguments raise,
and division additionally rejects a zero divisor. Registration connects a name
to a callable; importing this package does not execute or register a tool.
"""
