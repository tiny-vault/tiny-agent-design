"""The agent's tools.

* ``prompting.py``: ``PromptingTools``, the registry for a model that is told about its tools in the prompt.
* ``learned.py``: ``LearnedTools``, the registry for a model that learned tool calling in training.
* ``functions/``: the tool functions, one file each (``multiply.py``, ``divide.py``). ``default.py``: ``default_tools()`` and ``default_native_tools()``, which register ``multiply``.

Import from the files, for example ``from tiny_agent.components.tools.default import default_tools``.

A tool definition tells the model what it can request; the registered Python
function performs the operation. ``BaseTools`` shares dispatch, optional
approval and observation formatting. The prompting and learned variants differ
in how definitions are sent and calls parsed, not in the arithmetic they run.
An observation is returned after one action; the agent has no automatic loop.
"""
