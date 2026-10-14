"""Prompt-based tool use: describe functions and ask for a JSON action.

A model can choose a tool from instructions supplied with its conversation.
This implementation teaches the response format in the system prompt instead
of using the API's structured tool-call field. It does not train the model or
assume that every response will obey the requested format.

``PromptingTools.prompt`` lists the descriptions and requests
``{"tool": "name", "kwargs": {"param": "value"}}``. The inherited ``schemas``
stays ``None``, so the client sends no native definitions. ``parse`` looks for
a tool marker in the reply text and decodes the substring from its first
opening brace to its last closing brace. It copies content, reasoning and
metadata into a response with the parsed ``tool_call``. Text without that
marker is returned unchanged as an ordinary answer.

The base registry dispatches the parsed action. For suitable multiplication
arguments the observation is ``OBSERVATION: 42.0``; this illustrates the output
shape, not a guarantee that a model will choose correctly. Another run is
needed for a final model-written answer.

Malformed JSON raises, extra braces in prose can make extraction ambiguous,
and decoded data is not schema-validated. Compare ``LearnedTools``, which
reads the structured call instead of text.
"""

import json
from typing import override

from tiny_agent.components.client.response import Response
from tiny_agent.components.tools.tools import BaseTools


class PromptingTools(BaseTools):
    """Tool registry for the Agent: the model calls a tool by writing JSON in its reply."""

    @property
    @override
    def prompt(self) -> str:
        """The system-prompt section that tells the model which tools exist and how to call one."""
        return f"""
# Tools

If needed, you can only use the following tools to assist you in completing tasks:

{self.descriptions}

To use a tool, respond with JSON: {{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    @override
    def parse(self, response: Response) -> Response:
        """Parse a JSON tool call from the reply text.

        Returns:
            A copy of the response with ``tool_call`` set to ``{"tool", "kwargs"}``,
            or the response itself when the text has no recognized tool marker.

        Args:
            response: The reply whose content is scanned; it is not modified.

        Raises:
            ValueError: The brace-delimited substring is not valid JSON.

        The marker is a text heuristic, not a schema check or intent detector.
        """
        text = response.content or ""

        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            tool_call = json.loads(text[start:end])

            # Add the parsed tool call to the response
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
                metadata=response.metadata,
            )

        return response
