"""The date agent: a TinyAgent that specializes in dates, with Gemma 4 E4B thinking and calling tools natively.

A model does not know today's date (its knowledge stops at training) and often miscounts days across months. This
specialist hands both to code: its only tools are ``today`` (the computer's date as ``YYYY-MM-DD``) and
``days_between`` (the exact difference of two such dates). A question such as "how many days until Christmas?"
becomes ``today()``, then ``days_between(<today>, "<year>-12-25")``, then an answer.

``create_date_agent`` builds it like ``create_math_agent``: ``gemma4-e4b`` with ``think=True``, a ``LearnedTools``
registry (the book's ``NativeTools``) with each function's docstring as its description, and ``NativeReAct``. Each
call returns a new agent with an empty ``Memory``. The model server and its key come from ``Settings``;
``gemma4-e4b`` must be loaded (``node execute.mjs llm-client load gemma4-e4b``). The date is the machine's local
date, so in Docker it is the container's time zone. Compare ``math_agent.py``, the same agent with other tools.

Run ``python -m tiny_agent.agents.date_agent`` to ask it how many days are left until Christmas.
"""

import inspect
from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.functions.days_between import days_between
from tiny_agent.components.tools.functions.today import today
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.config.settings import Settings

MODEL = "gemma4-e4b"


def create_date_agent(llm: LLM | None = None) -> TinyAgent:
    """Create the date specialist: today and days_between as native tools, in a NativeReAct loop.

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

    # Date specialist
    date_tools = LearnedTools()
    date_tools.add_tool("today", today, inspect.getdoc(today) or "")
    date_tools.add_tool("days_between", days_between, inspect.getdoc(days_between) or "")

    # Date Agent
    date_agent = TinyAgent(
        llm=llm,
        tools=date_tools,
        memory=Memory(),
        planner=NativeReAct(),
    )
    return date_agent


if __name__ == "__main__":
    date_agent = create_date_agent()
    print(date_agent.run("How many days are left until Christmas this year?"))
    for index, step in enumerate(date_agent.trajectory.runs[0]["steps"]):
        print(f"-- Step {index + 1} ---")
        print(f"Tool: {step.action} -> {step.observation}" if step.action else f"Answer: {step.answer}")
