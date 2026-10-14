"""Native tool calling: supply runtime schemas and normalize structured calls.

A model trained to follow a tool-call protocol still needs definitions of the
functions available in this application. Here definitions travel in the
request's ``tools`` field. The server's chat template presents them to the
model; an empty ``prompt`` means this registry adds no tool-description text,
not that the model already knows these particular functions.

``schemas`` inspects each callable and builds an OpenAI-style definition
with its name, description and parameters. A parameter's type comes from its
annotation (``float`` is ``number``, ``int`` is ``integer``, ``bool`` is
``boolean``, anything else ``string``) and its description from the docstring's
``Args:`` entry; parameters without Python defaults are required. This teaching
implementation does not infer other Python types or validate every kind of
callable signature.

``parse`` reads the first structured call kept by ``LLM.generate``. Arguments
may be a JSON string or an already decoded object. They become the shared
``{"tool": name, "kwargs": arguments}`` shape. Replies without a call or
without its ``function`` member are returned unchanged. JSON written only in
the content text is not interpreted here.

The base registry handles execution and approval. A result becomes a user-role
observation; another agent run is needed for a textual answer. Malformed JSON,
missing function names and invalid callable arguments can raise; there is no
recovery or sandbox. Compare ``PromptingTools`` for the same dispatch path
driven by JSON in ordinary reply text.
"""

import inspect
import json
from collections.abc import Callable
from typing import Any, get_args, override

from tiny_agent.components.client.response import Response
from tiny_agent.components.tools.tools import BaseTools

JSON_TYPES = {float: "number", int: "integer", bool: "boolean"}


def json_type(annotation: Any) -> str:
    """The JSON-schema type for a parameter annotation: ``number``, ``integer``, ``boolean``, else ``string``.

    A union such as ``float | str`` (a number that text is also accepted for) counts as its first listed type
    with a JSON type; an unannotated parameter is a string.
    """
    for candidate in get_args(annotation) or (annotation,):
        if candidate in JSON_TYPES:
            return JSON_TYPES[candidate]
    return "string"


def parameter_docs(function: Callable[..., Any]) -> dict[str, str]:
    """The description of each parameter, read from the ``Args:`` (or ``Arguments:``) section of the docstring.

    Only the first line of an entry (``name: text``) is read. A function without such a section gives ``{}``.
    """
    docs: dict[str, str] = {}
    in_section = False
    for line in (inspect.getdoc(function) or "").splitlines():
        if line.rstrip() in ("Args:", "Arguments:"):
            in_section = True
        elif in_section and line.startswith(" ") and ":" in line:
            name, _, text = line.strip().partition(":")
            if line.startswith("    ") and not line.startswith("     "):
                docs[name.strip()] = text.strip()
        elif in_section and line.strip() and not line.startswith(" "):
            in_section = False
    return docs


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

        A parameter's JSON type comes from its annotation (``float`` is ``number``, ``int`` is
        ``integer``, ``bool`` is ``boolean``, anything else ``string``) and its description from the
        function's docstring ``Args:`` entry; a parameter without a default is required.
        """
        schemas = []
        for name, entry in self.registry.items():
            function = entry["function"]
            parameters = inspect.signature(function).parameters.values()
            docs = parameter_docs(function)
            properties = {}
            for p in parameters:
                properties[p.name] = {"type": json_type(p.annotation)}
                if p.name in docs:
                    properties[p.name]["description"] = docs[p.name]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": entry["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
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
            the response itself when the model made no structured function call.

        Args:
            response: The client reply; parsing does not execute its requested tool.

        Raises:
            ValueError: A string of arguments is not valid JSON.
            KeyError: The structured function has no name.
            TypeError: The function object has an incompatible shape.
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
