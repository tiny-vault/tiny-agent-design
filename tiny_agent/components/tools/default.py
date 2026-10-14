"""The tools the agent gets by default, registered in one place."""

from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.components.tools.multiply import multiply
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.components.tools.tools import BaseTools
from tiny_agent.config.settings import Settings


def default_tools(requires_approval: list[str] | None = None) -> PromptingTools:
    """Build a ``PromptingTools`` registry with the built-in tools registered.

    Arguments:
        requires_approval: Names of tools that ask the user before they run.
    """
    tools = PromptingTools(requires_approval=requires_approval or [])
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)",
    )
    return tools


def default_native_tools(requires_approval: list[str] | None = None) -> LearnedTools:
    """Build a ``LearnedTools`` registry (for a model that learned tool calling) with the built-in tools registered."""
    tools = LearnedTools(requires_approval=requires_approval or [])
    tools.add_tool(name="multiply", func=multiply, description="Multiplies two numbers: multiply(a: str, b: str)")
    return tools


def tools_from_settings(settings: Settings | None = None) -> BaseTools | None:
    """Bind the tools the flags ask for: ``LearnedTools`` for ``TINYAGENT_NATIVE_TOOLS``, ``PromptingTools`` for ``TINYAGENT_TOOLS``, else ``None``."""
    kind = (settings or Settings.from_env()).tools_kind
    if kind == "native":
        return default_native_tools()
    return default_tools() if kind == "prompt" else None
