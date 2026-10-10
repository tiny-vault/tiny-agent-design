"""Tests for the TinyAgent skeleton, the Response dataclass and the demo entry point."""

import dataclasses

from tiny_agent import main as demo
from tiny_agent.shared.response import Response
from tiny_agent.shared.tiny_agent import TinyAgent


# ---- TinyAgent (placeholders until the later steps) --------------------------------

def test_the_slots_are_empty_until_their_step():
    agent = TinyAgent()
    assert (agent.llm, agent.memory, agent.tools, agent.planner) == (None, None, None, None)


def test_run_echoes_the_task():
    assert TinyAgent().run("Hello, agent!") == "Received: Hello, agent!"


def test_run_handles_empty_and_unusual_text():
    assert TinyAgent().run("") == "Received: "
    assert TinyAgent().run("héllo 🌍\nline two") == "Received: héllo 🌍\nline two"


def test_step_and_execute_action_placeholders():
    agent = TinyAgent()
    assert agent._step("x") == "Received: x"
    assert agent._execute_action("search") == "Executed action: search"


# ---- Response ----------------------------------------------------------------------

def test_response_defaults():
    response = Response()
    assert (response.content, response.reasoning, response.tool_call, response.metadata) == ("", None, None, None)


def test_response_is_a_dataclass_with_the_four_fields():
    assert [f.name for f in dataclasses.fields(Response)] == ["content", "reasoning", "tool_call", "metadata"]


def test_response_equality_and_fields():
    a = Response(content="x", metadata={"model": "m"})
    assert a == Response(content="x", metadata={"model": "m"})
    assert a != Response(content="y", metadata={"model": "m"})


# ---- the demo entry point -----------------------------------------------------------

def test_main_prints_the_answer(capsys):
    """main() builds the agent, runs one task and prints the result; it does not call a model."""
    demo.main()
    assert capsys.readouterr().out.strip() == "Received: Hello, agent!"


def test_main_does_not_need_a_model_server(capsys, dead_url, monkeypatch):
    """The chat with a model lives in the tiny-agent CLI, so the demo works with nothing running."""
    monkeypatch.setenv("TINYAGENT_LLM_URL", dead_url)
    demo.main()
    assert "Received: Hello, agent!" in capsys.readouterr().out
