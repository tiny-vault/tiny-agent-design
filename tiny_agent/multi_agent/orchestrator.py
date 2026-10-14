"""The orchestrator: an agent whose tools are other agents, the math and the date specialist.

A question such as "If I save €4 per day until 2030, how much will I have?" needs two skills: counting the days
from today to 2030 (a date question) and multiplying by 4 (a math question). The orchestrator has no date or
arithmetic tools of its own. Its two tools are functions that pass a question to a specialist and return its
answer: ``ask_math_agent`` runs the math agent, ``ask_date_agent`` the date agent. So the orchestrator plans and
delegates, each specialist runs its own NativeReAct loop with its own tools, and its final answer comes back to the
orchestrator as an observation.

``create_orchestrator`` builds the book's setup: ``gemma4-e4b`` with native thinking, a ``LearnedTools`` registry
(the book's ``NativeTools``) holding the two ``ask_*`` functions, ``NativeReAct`` and an empty ``Memory``. The
model reads each tool's name and description, and each description is the function's docstring (passed to
``add_tool``, since ``LearnedTools`` does not read docstrings itself), so those docstrings tell it which specialist
to ask. The two
specialists are created once per orchestrator and kept, as in the book: their memory grows with every question the
orchestrator sends them, and each question is a new run in their own trajectory.

The orchestrator only sees the specialists' final answers, never their steps; look in a specialist's trajectory
for those. Each delegation is a full agent run, so a question costs several model calls. A specialist that fails
raises inside the tool, which the ReAct loop turns into an error observation. Compare ``tiny_agent/agents/`` for
the specialists.

With the ``a2a`` flag (``TINYAGENT_A2A=1``, or ``--a2a`` when run as a script) the specialists are not called in
the same process: each is served over the A2A protocol (``a2a.py``) on a free local port and the ``ask_*`` tools
send it a message through ``A2AClient``. The orchestrator then knows its specialists only by URL, the way it would
reach agents on another machine; the answers are the same. The servers live as long as the program.

Run ``python -m tiny_agent.multi_agent.orchestrator [--a2a]`` (needs ``gemma4-e4b`` loaded).
"""

import inspect
import sys
from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.agents.date_agent import create_date_agent
from tiny_agent.agents.math_agent import create_math_agent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.config.settings import Settings
from tiny_agent.multi_agent.a2a import A2AClient, A2AServer

MODEL = "gemma4-e4b"


def create_orchestrator(
    llm: LLM | None = None, math_agent: TinyAgent | None = None, date_agent: TinyAgent | None = None,
    a2a: bool | None = None,
) -> TinyAgent:
    """Create the orchestrator, with the math and the date specialist as its two tools.

    Args:
        llm: The model for the orchestrator and, unless given, both specialists. Defaults to ``gemma4-e4b``
            with native thinking, on the server ``Settings`` names.
        math_agent: The math specialist. Defaults to ``create_math_agent(llm)``.
        date_agent: The date specialist. Defaults to ``create_date_agent(llm)``.
        a2a: Reach the specialists over A2A instead of in-process. Defaults to ``Settings.a2a`` (``TINYAGENT_A2A``).

    Returns:
        A new orchestrator ``TinyAgent`` with its own empty ``Memory``.
    """
    if llm is None:
        settings = Settings.from_env()
        api_key = settings.resolve_api_key()
        key: dict[str, Any] = {"api_key": api_key} if api_key else {}
        # Gemma 4 E4B (with native thinking and tool calling)
        llm = LLM(model=MODEL, base_url=settings.llm_url, think=True, **key)
    math_specialist = math_agent or create_math_agent(llm)
    date_specialist = date_agent or create_date_agent(llm)
    if a2a is None:
        a2a = Settings.from_env().a2a
    math_remote = date_remote = None
    if a2a:  # serve each specialist and keep only a client for it: from here on they are just URLs
        math_remote = A2AClient(A2AServer(math_specialist, "Math agent", "Arithmetic with add, subtract and multiply.").start().url)
        date_remote = A2AClient(A2AServer(date_specialist, "Date agent", "Today's date and days between ISO dates.").start().url)

    # The specialists as functions: the name and the docstring (passed below as the description) are what the
    # orchestrator reads to choose one
    def ask_math_agent(question: str) -> str:
        """Delegate to the math specialist."""
        return math_remote.send(question) if math_remote else math_specialist.run(question)

    def ask_date_agent(question: str) -> str:
        """Ask queries to a sub-agent that handles ISO dates."""
        return date_remote.send(question) if date_remote else date_specialist.run(question)

    # Tools - Add the Math Agent and Date Agent as tools!
    tools = LearnedTools()
    tools.add_tool("ask_math_agent", ask_math_agent, inspect.getdoc(ask_math_agent) or "")
    tools.add_tool("ask_date_agent", ask_date_agent, inspect.getdoc(ask_date_agent) or "")

    # Orchestrator Agent
    orchestrator_agent = TinyAgent(
        llm=llm,
        tools=tools,
        memory=Memory(),
        planner=NativeReAct(),
    )
    return orchestrator_agent


if __name__ == "__main__":
    # --model <name> runs the orchestrator and both specialists on another loaded model (e.g. qwen3.5-4b)
    chosen = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else MODEL
    settings = Settings.from_env()
    api_key = settings.resolve_api_key()
    llm = LLM(model=chosen, base_url=settings.llm_url, think=True, **({"api_key": api_key} if api_key else {}))
    orchestrator_agent = create_orchestrator(llm, a2a=True if "--a2a" in sys.argv else None)
    print(orchestrator_agent.run("If I save €4 per day until 2030, how much will I have?"))
    print()
    for index, step in enumerate(orchestrator_agent.trajectory.runs[0]["steps"]):
        print(f"-- Step {index+1} ---")
        if step.action:
            print(f"Tool: {step.action}")
            print(f"Observation: {step.observation}\n")
        else:
            print(f"Answer: {step.answer}\n")
