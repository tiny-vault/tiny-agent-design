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
            or the response itself when the text holds no tool call.
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
