"""Default tool wiring: choose a registry without changing the agent.

A registry must hold actual callables before a model can request them.
``default_tools`` registers ``multiply``, ``add``, ``subtract`` and ``divide`` with ``PromptingTools``;
``default_native_tools`` registers the same functions with ``LearnedTools``.
The prompting registry describes it as ``multiply(a: str, b: str)`` in the prompt; the native one gives the
schema ``number`` parameters taken from the function. Both can receive a list
of names that need approval. They construct fresh registries and make no
model calls; execution happens later through the agent.

``tools_from_settings`` reads ``Settings.tools_kind``: native tools win when
both flags are on, prompt tools are selected by ``TINYAGENT_TOOLS`` alone,
and neither flag returns ``None`` for plain chat. The native flag is
``TINYAGENT_NATIVE_TOOLS``. Multiplication, addition, subtraction and division are registered by default;
``divide`` exists in ``functions/`` but is registered by the happy-path demo.

The returned object controls how definitions reach the model, not whether the
model will choose a correct call. Compare ``prompting.py`` and ``learned.py``
for those formats. No tools are executed simply by creating this registry.
"""

from tiny_agent.components.tools.functions.add import add
from tiny_agent.components.tools.functions.divide import divide
from tiny_agent.components.tools.functions.multiply import multiply
from tiny_agent.components.tools.functions.subtract import subtract
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.components.tools.tools import BaseTools
from tiny_agent.config.settings import Settings


def default_tools(requires_approval: list[str] | None = None) -> PromptingTools:
    """Build a ``PromptingTools`` registry with the built-in tools registered.

    Args:
        requires_approval: Names of tools that ask the user before they run.

    Returns:
        A fresh prompting registry containing ``multiply``.
    """
    tools = PromptingTools(requires_approval=requires_approval or [])
    tools.add_tool(
        name="multiply",
        func=multiply,
        description=(
            "Multiplies two numbers and returns the product as a decimal number. Use it for every "
            "multiplication instead of working it out yourself. Both arguments are numbers written "
            "as text, such as \"6\"; text that is not a number is an error: multiply(a: str, b: str). "
            "Example: multiply(\"6\", \"7\") returns 42.0."
        ),
    )
    tools.add_tool("add", add, "Adds two numbers and returns the sum: add(a: str, b: str). Example: add(\"4\", \"3\") returns 7.0.")
    tools.add_tool(
        "subtract", subtract,
        "Subtracts b from a and returns the difference: subtract(a: str, b: str). Example: subtract(\"9\", \"2\") returns 7.0.",
    )
    tools.add_tool(
        "divide", divide,
        "Divides a by b and returns the quotient; b must not be zero: divide(a: str, b: str). Example: divide(\"42\", \"6\") returns 7.0.",
    )
    return tools


def default_native_tools(requires_approval: list[str] | None = None) -> LearnedTools:
    """Build a fresh native registry containing the built-in ``multiply`` tool.

    Args:
        requires_approval: Registered names that ask through ``input`` before execution.

    Returns:
        A ``LearnedTools`` registry; definitions are generated when schemas are read.
    """
    tools = LearnedTools(requires_approval=requires_approval or [])
    # The schema's parameter types and descriptions come from multiply's annotations and docstring
    tools.add_tool(
        name="multiply",
        func=multiply,
        description=(
            "Multiplies two numbers and returns the product as a decimal number. Use it for every "
            "multiplication instead of working it out yourself. Text that is not a number is an error. "
            "Example: a=6, b=7 returns 42.0."
        ),
    )
    tools.add_tool("add", add, "Adds two numbers and returns the sum. Example: a=4, b=3 returns 7.0.")
    tools.add_tool("subtract", subtract, "Subtracts b from a and returns the difference. Example: a=9, b=2 returns 7.0.")
    tools.add_tool("divide", divide, "Divides a by b and returns the quotient; b must not be zero. Example: a=42, b=6 returns 7.0.")
    return tools


def tools_from_settings(settings: Settings | None = None) -> BaseTools | None:
    """Bind the tools the flags ask for: ``LearnedTools`` for ``TINYAGENT_NATIVE_TOOLS``, ``PromptingTools`` for ``TINYAGENT_TOOLS``, else ``None``."""
    kind = (settings or Settings.from_env()).tools_kind
    if kind == "native":
        return default_native_tools()
    return default_tools() if kind == "prompt" else None
