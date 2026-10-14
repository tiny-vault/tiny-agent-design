"""Tool dispatch: turn a model's requested action into a Python function call.

A language model can request an action but does not execute the function itself.
The registry is the harness's address book: each name maps to a callable and a
description. Separating dispatch from parsing lets different model formats use
the same execution and approval rules.

``BaseTools`` supplies ``add_tool``, ``execute``, ``observation`` and ``is_done``.
A subclass supplies ``prompt`` and ``parse``; native calling also supplies
``schemas``. Parsed calls use ``{"tool": name, "kwargs": arguments}``; ``kwargs``
is expanded as keyword arguments into the registered function. Registering the
same name again replaces its entry.

Execution asks through ``input`` only for registered names listed in
``requires_approval``. Only ``y`` or ``yes`` permits execution. Denial and an
unknown name return explanatory text rather than raising. A successful call
returns the function's own result; ``observation`` wraps any result as a
``user`` message containing ``OBSERVATION: <result>`` (``LearnedTools`` uses the
``tool`` role for the same text).

``is_done`` means no tool call was requested, or the model called ``final_answer``
(its ``kwargs`` become the answer), not that a task was verified.
This registry does not validate schemas, catch callable errors, sandbox tools,
or run a model loop. Bad parsed shapes and callable errors propagate.
Approval is an optional per-name check, not a general security boundary.
Compare ``PromptingTools`` and ``LearnedTools`` for the two parsing formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from tiny_agent.components.client.response import Response


class BaseTools(ABC):
    """What every tool registry has: the registered tools and how to run one.

    How the model is told about the tools, how its tool call is read and what
    counts as done differ between registries, so a subclass such as
    :class:`tiny_agent.components.tools.prompting.PromptingTools` provides ``prompt`` and ``parse``
    (and ``schemas``, when the tools travel in the request instead of the prompt). The observation
    and the stopping rule are the same for both and live here.
    """

    def __init__(self, requires_approval: list[str] | None = None):
        """Start an empty registry and select names that require approval.

        Args:
            requires_approval: Names checked at execution; ``None`` creates a
                fresh list. A supplied list is retained rather than copied.
        """
        self.registry = {}
        self.requires_approval = requires_approval if requires_approval is not None else []

    def add_tool(self, name: str, func: Callable[..., Any], description: str = "") -> None:
        """Register a tool that the Agent can use.

        Args:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
        """
        self.registry[name] = {"function": func, "description": description}

    @property
    def descriptions(self) -> str:
        """Get descriptions of all registered tools, one per line."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}" for tool in self.registry
        )

    @property
    @abstractmethod
    def prompt(self) -> str:
        """The system-prompt section that tells the model which tools exist."""

    @abstractmethod
    def parse(self, response: Response) -> Response:
        """Read a tool call out of the model's reply."""

    def execute(self, response: Response) -> Any:
        """Run the registered tool the response asks for.

        Args:
            response: A parsed response whose ``tool_call`` has "tool" and "kwargs" keys.

        Returns:
            The callable result, or text describing denial or an unknown name.

        Raises:
            KeyError: The parsed call has no tool name.
            TypeError: Arguments cannot be expanded or do not match the callable.
            EOFError: Approval input ends before an answer.

        Callable exceptions also propagate; no retry or result validation is added.
        """
        tool_call = response.tool_call or {}
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # Approval is opt-in per name; ordinary registration does not ask.
        if name in self.registry and name in self.requires_approval:
            answer = input(f"Allow {name}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # Dispatch only names explicitly present in this registry.
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."

    def observation(self, result: Any) -> tuple[str, str]:
        """Return the tool's result as a user message, so the model reads it on the next step."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """The agent's stopping mechanism: a reply with no tool call, or a ``final_answer`` call, is the answer.

        A ``final_answer`` call is how the model says it is finished: its ``kwargs`` become the reply's
        ``content`` (a string as is, anything else as ``str``). Nothing registers a ``final_answer`` tool; the
        model has to be told about it in the prompt.
        """
        if not response.tool_call:
            return True
        if response.tool_call.get("tool") == "final_answer":
            answer = response.tool_call.get("kwargs", "")
            response.content = answer if isinstance(answer, str) else str(answer)
            return True
        return False

    @property
    def schemas(self) -> None:
        """No native definitions by default; ``LearnedTools`` supplies them."""
        return None
