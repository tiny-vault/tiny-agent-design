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
        """Initialize and select tools that require approval before execution.
        """
        self.registry = {}
        self.requires_approval = requires_approval if requires_approval is not None else []

    def add_tool(self, name: str, func: Callable[..., Any], description: str = "") -> None:
        """Register a tool that the Agent can use.

        Arguments:
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

        Arguments:
            response: A parsed response whose ``tool_call`` has "tool" and "kwargs" keys.
        """
        tool_call = response.tool_call or {}
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # Human-in-the-loop: ask before running dangerous tools
        if name in self.registry and name in self.requires_approval:
            answer = input(f"Allow {name}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # Handle registered tools
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."

    def observation(self, result: Any) -> tuple[str, str]:
        """Return the tool's result as a user message, so the model reads it on the next step."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """The agent's stopping mechanism: a reply with no tool call is the answer."""
        return not response.tool_call

    @property
    def schemas(self) -> None:
        """Used only for native tool-calling."""
        return None
