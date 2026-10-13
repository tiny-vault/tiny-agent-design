"""Tests for the TinyAgent skeleton, the Response dataclass and the demo entry point.

These tests never need a real model. ``EchoLLM`` replies "Received: <last message>",
so a test can assert exactly what the agent sent and what it returned. The demo tests
use the fake server from ``conftest.py`` instead, to cover the real HTTP path.
"""

import dataclasses
from typing import Any

import pytest

import main as demo
from llm_client.llm import LLM
from llm_client.response import Response
from tiny_agent.shared.memory import Memory
from tiny_agent.shared.tiny_agent import TinyAgent


class EchoLLM(LLM):
    """A stand-in for ``LLM``: answers ``"Received: <last message>"`` and records every call.

    A subclass so it type-checks where ``TinyAgent`` expects an ``LLM``; it never touches the network.
    """

    def __init__(self):
        super().__init__(model="echo")
        self.calls = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        # A copy: memory hands out its live list, which grows after the call returns.
        self.calls.append(list(messages))
        return Response(content=f"Received: {messages[-1]['content']}")


# ---- TinyAgent (one model call per task, remembered in memory; tools and planner are placeholders) ----

def test_the_llm_and_memory_are_kept_and_the_other_slots_are_empty_until_their_step():
    llm, memory = EchoLLM(), Memory()
    agent = TinyAgent(llm, memory)
    assert agent.llm is llm
    assert agent.memory is memory
    assert (agent.tools, agent.planner) == (None, None)


def test_run_returns_the_models_reply():
    assert TinyAgent(EchoLLM(), Memory()).run("Hello, agent!") == "Received: Hello, agent!"


def test_run_sends_the_task_as_one_user_message():
    llm = EchoLLM()
    TinyAgent(llm, Memory()).run("Hello, agent!")
    assert llm.calls == [[{"role": "user", "content": "Hello, agent!"}]]


def test_run_remembers_the_task_and_the_reply():
    memory = Memory()
    TinyAgent(EchoLLM(), memory).run("Hello, agent!")
    assert memory.get_messages() == [
        {"role": "user", "content": "Hello, agent!"},
        {"role": "assistant", "content": "Received: Hello, agent!"},
    ]


def test_a_second_run_sends_the_first_exchange_too():
    llm = EchoLLM()
    agent = TinyAgent(llm, Memory())
    agent.run("one")
    agent.run("two")
    assert llm.calls[1] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "Received: one"},
        {"role": "user", "content": "two"},
    ]


def test_run_handles_empty_and_unusual_text():
    assert TinyAgent(EchoLLM(), Memory()).run("") == "Received: "
    assert TinyAgent(EchoLLM(), Memory()).run("héllo 🌍\nline two") == "Received: héllo 🌍\nline two"


def test_step_and_execute_action_placeholders():
    agent = TinyAgent(EchoLLM(), Memory())
    agent.trajectory.initialize("x")
    agent.memory.add("user", "x")
    assert agent._step() == "Received: x"
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
    """main() runs the greeting and then the question, and prints the second answer."""
    demo.main()
    assert capsys.readouterr().out.strip() == "Your names are Maarten and Jay."
    assert demo_server.requests[0]["body"]["messages"] == [{"role": "user", "content": demo.GREETING}]


def test_main_sends_the_greeting_again_with_the_question(demo_server):
    """Memory: the second request carries the greeting and its reply, then the question."""
    demo.main()
    assert demo_server.requests[1]["body"]["messages"] == [
        {"role": "user", "content": demo.GREETING},
        {"role": "assistant", "content": "Your names are Maarten and Jay."},
        {"role": "user", "content": demo.QUESTION},
    ]


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
