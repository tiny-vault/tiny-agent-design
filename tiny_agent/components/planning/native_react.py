"""Native ReAct: the same think, act, observe loop, with thinking and tool calls the model does natively.

``ReAct`` (``react.py``) teaches the cycle through text: the system prompt asks for a
``THOUGHT:`` and an ``ACTION:`` as JSON, and ``ReAct.parse`` splits the reply apart. A model
trained to reason and to call tools needs neither. With ``LLM(think=True)`` the server returns
its thinking separately, in ``Response.reasoning``; with ``LearnedTools`` the tool schemas travel
in the request and the call comes back as a structured ``tool_call``. So ``NativeReAct`` has an
empty ``prompt`` and a ``parse`` that changes nothing, and ``TinyAgent.run`` loops exactly as before.

The loop ends on a reply with no tool call (``BaseTools.is_done``). ``final_answer`` is described
only in ``ReAct.prompt`` and is not among the schemas a learned registry offers, so a native model
finishes by answering in plain text. The trajectory's thought is whatever native thinking the server
returned; with ``think=False`` it is empty. Compare ``react.py`` for the text-based version.
"""

from typing import override

from tiny_agent.components.client.response import Response
from tiny_agent.components.planning.react import ReAct


class NativeReAct(ReAct):
    """ReAct using native LLM reasoning instead of text-based parsing."""

    @property
    @override
    def prompt(self) -> str:
        """Nothing: thinking and tool calls are native, so no THOUGHT / ACTION format is taught."""
        return ""

    @override
    def parse(self, response: Response) -> Response:
        """Return the reply unchanged: the thinking is already in ``reasoning`` and the call in ``tool_call``."""
        return response
