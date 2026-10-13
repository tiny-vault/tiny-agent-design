"""Tests for the TinyAgent skeleton, the Response dataclass and the demo entry point."""

import dataclasses

import pytest

import main as demo
from llm_client.response import Response
from tiny_agent.shared.tiny_agent import TinyAgent


class EchoLLM:
    """A stand-in for ``LLM``: answers ``"Received: <last message>"`` and records every call."""

    def __init__(self):
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append(messages)
        return Response(content=f"Received: {messages[-1]['content']}")


# ---- TinyAgent (one model call per task; the other slots are placeholders) ----------

def test_the_llm_is_kept_and_the_other_slots_are_empty_until_their_step():
    llm = EchoLLM()
    agent = TinyAgent(llm)
    assert agent.llm is llm
    assert (agent.memory, agent.tools, agent.planner) == (None, None, None)


def test_run_returns_the_models_reply():
    assert TinyAgent(EchoLLM()).run("Hello, agent!") == "Received: Hello, agent!"


def test_run_sends_the_task_as_one_user_message():
    llm = EchoLLM()
    TinyAgent(llm).run("Hello, agent!")
    assert llm.calls == [[{"role": "user", "content": "Hello, agent!"}]]


def test_run_handles_empty_and_unusual_text():
    assert TinyAgent(EchoLLM()).run("") == "Received: "
    assert TinyAgent(EchoLLM()).run("héllo 🌍\nline two") == "Received: héllo 🌍\nline two"


def test_step_and_execute_action_placeholders():
    agent = TinyAgent(EchoLLM())
    agent.trajectory.initialize("x")
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

def test_main_prints_the_answer(demo_server, capsys):
    """main() builds the agent, runs one task through the model and prints the result."""
    demo.main()
    assert capsys.readouterr().out.strip() == "Received: Hello, agent!"
    assert demo_server.requests[0]["body"]["messages"] == [{"role": "user", "content": "Hello, agent!"}]


def test_main_uses_the_model_named_in_the_environment(demo_server, monkeypatch):
    demo_server.router(other="loaded")
    monkeypatch.setenv("TINYAGENT_MODEL", "other")
    demo.main()
    assert demo_server.requests[0]["body"]["model"] == "other"


def test_main_needs_a_model_server(dead_url, monkeypatch):
    """The agent now calls the model, so the demo fails clearly when nothing is listening."""
    monkeypatch.setenv("TINYAGENT_LLM_URL", dead_url)
    with pytest.raises(RuntimeError, match="Cannot reach the model"):
        demo.main()
