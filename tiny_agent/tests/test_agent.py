"""Tests for the TinyAgent skeleton, the Response dataclass and the demo entry point."""

import dataclasses

import pytest

from tiny_agent import main as demo
from tiny_agent.shared.llm import LLM
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

def test_main_prints_the_agent_answer_and_the_models_reply(capsys, fake_server, monkeypatch):
    """main() runs the agent, then asks the model "Hi! How's life?" and prints the Response."""
    fake_server.router(**{"gemma4:e4b": "loaded"})
    fake_server.reply("Doing great, thanks!")
    monkeypatch.setattr(demo, "LLM", lambda model: LLM(model=model, base_url=fake_server.url))

    demo.main()

    out = capsys.readouterr().out
    assert out.splitlines()[0] == "Received: Hello, agent!"
    assert "Doing great, thanks!" in out and out.splitlines()[1].startswith("Response(")
    assert fake_server.requests[0]["body"]["model"] == "gemma4:e4b"
    assert fake_server.requests[0]["body"]["messages"] == [{"role": "user", "content": "Hi! How's life?"}]


def test_main_fails_clearly_when_the_model_name_is_unknown(fake_server, monkeypatch):
    """The demo uses the Ollama tag gemma4:e4b; a server that does not offer it says so."""
    fake_server.router(**{"gemma4-e4b": "loaded"})
    fake_server.chat_status = 404
    fake_server.chat = {"error": {"message": "Unknown model: gemma4:e4b"}}
    monkeypatch.setattr(demo, "LLM", lambda model: LLM(model=model, base_url=fake_server.url))
    with pytest.raises(RuntimeError, match="Unknown model: gemma4:e4b"):
        demo.main()
