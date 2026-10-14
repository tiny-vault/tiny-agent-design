"""The agent's tools.

* ``prompting.py``: ``PromptingTools``, the registry for a model that is told about its tools in the prompt.
* ``learned.py``: ``LearnedTools``, the registry for a model that learned tool calling in training.
* ``multiply.py``: the example tool. ``default.py``: ``default_tools()``, which registers it.

Import from the files, for example ``from tiny_agent.components.tools.default import default_tools``.
"""
