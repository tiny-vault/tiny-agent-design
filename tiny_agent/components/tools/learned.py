import inspect
import json
from typing import Any, override

from tiny_agent.components.client.response import Response
from tiny_agent.components.tools.tools import BaseTools


class LearnedTools(BaseTools):
    """Tool registry for a model that learned tool calling during training.

    There are two ways to tell a model about its tools. ``PromptingTools`` *prompts* it: the
    definitions are written into the system prompt and the model answers with JSON in
    its text. Here the model *learned* the tool-call format in training, so the prompt
    holds no tool text at all. The definitions travel in the request's ``tools`` field
    (:attr:`schemas`), the server's chat template renders them the way the model was
    trained to read them, and the call comes back as a structured ``tool_call``.
    """

    @property
    @override
    def prompt(self) -> str:
        """Nothing: the model knows how to call a tool, and the definitions go in ``schemas``."""
        return ""

    @property
    @override
    def schemas(self) -> list[dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """The registered tools in the OpenAI function format, for the request's ``tools`` field.

        Every parameter is a string, like the prompt variant's ``multiply(a: str, b: str)``;
        a parameter without a default is required.
        """
        schemas = []
        for name, entry in self.registry.items():
            parameters = inspect.signature(entry["function"]).parameters.values()
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": entry["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {p.name: {"type": "string"} for p in parameters},
                            "required": [p.name for p in parameters if p.default is p.empty],
                        },
                    },
                }
            )
        return schemas

    @override
    def parse(self, response: Response) -> Response:
        """Turn the server's structured tool call into ``{"tool", "kwargs"}``.

        Returns:
            A copy of the response with ``tool_call`` set to ``{"tool", "kwargs"}``, or
            the response itself when the model made no tool call.
        """
        call = response.tool_call
        if not call or "function" not in call:
            return response

        function = call["function"]
        arguments = function.get("arguments") or {}
        # The server sends the arguments as a JSON string; some send an object already
        kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments

        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call={"tool": function["name"], "kwargs": kwargs},
            metadata=response.metadata,
        )
