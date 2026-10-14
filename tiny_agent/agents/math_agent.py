"""The math agent: a TinyAgent that specializes in arithmetic, with Gemma 4 E4B thinking and calling tools natively.

A *specialist* is an ordinary ``TinyAgent`` given a narrow job: here the only tools are ``add``, ``subtract`` and
``multiply``, so every task it gets is arithmetic done step by step through tool calls. Several specialists can later
be combined, each handling the part of a task it is built for.

``create_math_agent`` builds it the book's way: ``gemma4-e4b`` with ``think=True`` (the server returns the model's
own reasoning in ``Response.reasoning``, which the trajectory records as each step's thought), a ``LearnedTools``
registry (the book's ``NativeTools``: tool schemas in the request, structured calls back) and ``NativeReAct`` (no
THOUGHT / ACTION format to teach, the loop in ``TinyAgent.run`` does the rest). Each tool's docstring is passed as its
description: the book's ``NativeTools`` reads it by itself, ``LearnedTools`` sends only the description it is given.

Each call returns a new agent with an empty ``Memory``. The model server and its key come from ``Settings``
(``TINYAGENT_LLM_URL``, default the Docker gateway on port 3200); ``gemma4-e4b`` must be loaded there
(``node execute.mjs llm-client load gemma4-e4b``). Creating the agent sends nothing; the first ``run`` does.
Thinking makes each step slower than with ``think=False``. Compare ``tiny_agent/complete/evaluation.py``, which builds the
same kind of agent with no tools.

Run ``python -m tiny_agent.agents.math_agent`` to give it the book's multi-step task.
"""

import inspect
from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.functions.add import add
from tiny_agent.components.tools.functions.multiply import multiply
from tiny_agent.components.tools.functions.subtract import subtract
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.config.settings import Settings

MODEL = "gemma4-e4b"


def create_math_agent(llm: LLM | None = None) -> TinyAgent:
    """Create the math specialist: add, subtract and multiply as native tools, in a NativeReAct loop.

    Args:
        llm: The model. Defaults to ``gemma4-e4b`` with native thinking, on the server ``Settings`` names.

    Returns:
        A new ``TinyAgent`` with its own empty ``Memory``.
    """
    if llm is None:
        settings = Settings.from_env()
        api_key = settings.resolve_api_key()
        key: dict[str, Any] = {"api_key": api_key} if api_key else {}
        # Gemma 4 E4B (with native thinking and tool calling)
        llm = LLM(model=MODEL, base_url=settings.llm_url, think=True, **key)

    # Math specialist
    math_tools = LearnedTools()
    math_tools.add_tool("add", add, inspect.getdoc(add) or "")
    math_tools.add_tool("subtract", subtract, inspect.getdoc(subtract) or "")
    math_tools.add_tool("multiply", multiply, inspect.getdoc(multiply) or "")

    # Math Agent
    math_agent = TinyAgent(
        llm=llm,
        tools=math_tools,
        memory=Memory(),
        planner=NativeReAct(),
    )
    return math_agent


if __name__ == "__main__":
    math_agent = create_math_agent()
    print(math_agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?"))
    for index, step in enumerate(math_agent.trajectory.runs[0]["steps"]):
        print(f"-- Step {index + 1} ---")
        print(f"Thought: {step.thought[:200]}")
        print(f"Tool: {step.action}" if step.action else f"Answer: {step.answer}")
