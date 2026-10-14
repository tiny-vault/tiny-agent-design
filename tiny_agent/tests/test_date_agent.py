"""Tests for the date agent and its two tools: today and days_between, native calls in a NativeReAct loop."""

import datetime

import pytest

from tiny_agent.agents.date_agent import MODEL, create_date_agent
from tiny_agent.components.client.response import Response
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.functions.days_between import days_between
from tiny_agent.components.tools.functions.today import today
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.tests.stubs import ScriptedLLM


def call(name: str, arguments: str) -> dict[str, object]:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_today_is_the_local_date_in_iso_form():
    assert today() == datetime.date.today().isoformat()


def test_days_between_counts_calendar_days_and_can_be_negative():
    assert days_between("2026-10-15", "2026-12-25") == 71
    assert days_between("2028-02-28", "2028-03-01") == 2  # a leap year
    assert days_between("2026-12-25", "2026-10-15") == -71
    with pytest.raises(ValueError):
        days_between("Christmas", "2026-12-25")


def test_default_llm_is_gemma_with_native_thinking():
    agent = create_date_agent()
    assert (agent.llm.model, agent.llm.think, MODEL) == ("gemma4-e4b", True, "gemma4-e4b")


def test_date_agent_has_two_native_tools_with_typed_schemas():
    agent = create_date_agent(ScriptedLLM())
    assert isinstance(agent.tools, LearnedTools) and isinstance(agent.planner, NativeReAct)
    assert agent.tools.schemas is not None
    today_schema, between_schema = (schema["function"] for schema in agent.tools.schemas)
    assert (today_schema["name"], today_schema["parameters"]["required"]) == ("today", [])
    assert (between_schema["name"], between_schema["parameters"]["required"]) == ("days_between", ["a", "b"])


def test_date_agent_reads_the_date_then_counts_the_days():
    llm = ScriptedLLM(
        Response(content=None, tool_call=call("today", "{}")),
        Response(content=None, tool_call=call("days_between", '{"a": "2026-10-15", "b": "2026-12-25"}')),
        Response(content="71 days until Christmas."),
    )
    agent = create_date_agent(llm)
    assert agent.run("How many days until Christmas?") == "71 days until Christmas."
    first, second, answer = agent.trajectory.runs[0]["steps"]
    assert first.observation == f"OBSERVATION: {today()}"
    assert second.observation == "OBSERVATION: 71" and answer.answer == "71 days until Christmas."
