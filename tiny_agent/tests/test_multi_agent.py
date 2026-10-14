"""Tests for the orchestrator: two specialists registered as tools, each question delegated and answered."""

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.multi_agent.orchestrator import MODEL, create_orchestrator
from tiny_agent.tests.stubs import ScriptedLLM


def call(name: str, question: str) -> dict[str, object]:
    arguments = '{"question": "' + question + '"}'
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def specialist(*replies: str) -> TinyAgent:
    """A specialist that answers each question with the next reply, in plain text."""
    return TinyAgent(ScriptedLLM(*replies), Memory(), LearnedTools(), NativeReAct())


def test_default_llm_is_gemma_with_native_thinking_for_all_three_agents():
    agent = create_orchestrator()
    assert (agent.llm.model, agent.llm.think, MODEL) == ("gemma4-e4b", True, "gemma4-e4b")


def test_the_two_tools_are_the_specialists_described_by_their_docstrings():
    agent = create_orchestrator(ScriptedLLM())
    assert isinstance(agent.tools, LearnedTools) and agent.tools.schemas is not None
    functions = [schema["function"] for schema in agent.tools.schemas]
    assert [f["name"] for f in functions] == ["ask_math_agent", "ask_date_agent"]
    assert [f["description"] for f in functions] == [
        "Delegate to the math specialist.",
        "Ask queries to a sub-agent that handles ISO dates.",
    ]
    assert all(f["parameters"]["required"] == ["question"] for f in functions)


def test_the_orchestrator_delegates_to_each_specialist_and_combines_the_answers():
    date_agent = specialist("1174 days until 2030-01-01.")
    math_agent = specialist("4696")
    llm = ScriptedLLM(
        Response(content=None, tool_call=call("ask_date_agent", "Days from today until 2030-01-01?")),
        Response(content=None, tool_call=call("ask_math_agent", "What is 1174 times 4?")),
        Response(content="You will have EUR 4696."),
    )
    orchestrator = create_orchestrator(llm, math_agent=math_agent, date_agent=date_agent)
    assert orchestrator.run("If I save 4 per day until 2030, how much will I have?") == "You will have EUR 4696."
    # Each specialist got the orchestrator's question as its own task, and its answer came back as an observation
    assert date_agent.trajectory.runs[0]["query"] == "Days from today until 2030-01-01?"
    assert math_agent.trajectory.runs[0]["query"] == "What is 1174 times 4?"
    first, second, _ = orchestrator.trajectory.runs[0]["steps"]
    assert (first.observation, second.observation) == ("OBSERVATION: 1174 days until 2030-01-01.", "OBSERVATION: 4696")


def test_the_specialists_are_kept_between_questions():
    math_agent = specialist("2", "3")
    llm = ScriptedLLM(
        Response(content=None, tool_call=call("ask_math_agent", "1 + 1?")),
        Response(content="2"),
        Response(content=None, tool_call=call("ask_math_agent", "1 + 2?")),
        Response(content="3"),
    )
    orchestrator = create_orchestrator(llm, math_agent=math_agent, date_agent=specialist())
    assert [orchestrator.run("first"), orchestrator.run("second")] == ["2", "3"]
    assert [run["query"] for run in math_agent.trajectory.runs] == ["1 + 1?", "1 + 2?"]
