"""Tests for the TinyAgent harness: one model call per task, what memory carries from run to run, the
Step record, and the demo entry point.
"""

from typing import Any

import pytest

from tiny_agent import main as demo
from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.step import Step
from tiny_agent.components.trajectory import Trajectory
from tiny_agent.tests.stubs import GREETING, QUESTION, EchoLLM, NamesLLM

# ====================================================================================================
# test_agent
# One-decision agents and deterministic contract tests.
#
# The harness stores a task and reply so the next request can carry history.
# A test double supplies a known response, making the request history and returned
# value observable without judging model intelligence.
#
# What this file demonstrates:
# Tests for the TinyAgent one-decision harness, the Response model and the demo entry point.
#
# These tests never need a real model. ``EchoLLM`` replies "Received: <last message>",
# so a test can assert exactly what the agent sent and what it returned. The demo tests
# use the fake server from ``conftest.py`` instead, to cover the real HTTP path.
#
# Expected observations and limits:
# Observe the message order and the one call made by each run. Empty content
# becomes empty text for the no-tools agent. These assertions check wiring and
# data contracts, not whether a real model remembers accurately.
# ====================================================================================================

# ---- Plain-chat path: one decision per task; no registry is supplied in these tests ----

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


def test_a_reply_without_text_is_stored_and_returned_as_empty_text():
    """A tool-call-only reply has ``content=None``; with no tool registry supplied it counts as empty text."""

    class NoTextLLM(EchoLLM):
        def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
            return Response(content=None, tool_call={"id": "1"})

    memory = Memory()
    assert TinyAgent(NoTextLLM(), memory).run("hi") == ""
    assert memory.get_messages()[-1] == {"role": "assistant", "content": ""}


def test_step_without_tools_returns_the_reply():
    agent = TinyAgent(EchoLLM(), Memory())
    agent.trajectory.initialize("x")
    agent.memory.add("user", "x")
    assert agent._step() == "Received: x"


# ---- Response ----------------------------------------------------------------------

def test_response_defaults():
    response = Response()
    assert (response.content, response.reasoning, response.tool_call, response.metadata) == ("", None, None, None)


def test_response_has_the_four_fields():
    assert list(Response.model_fields) == ["content", "reasoning", "tool_call", "metadata"]


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


# ====================================================================================================
# test_memory_across_runs
# Shared versus fresh memory: isolate the effect of request history.
#
# A stateless model can use prior conversation only when its caller sends it.
# These two groups keep the tasks and stub behavior the same while changing
# whether one memory is reused or a new memory is constructed per run.
#
# What this file demonstrates:
# Checks that a ``TinyAgent`` WITH memory carries earlier turns into later ones, and one WITHOUT forgets.
#
# ``NamesLLM`` (``stubs.py``) uses a deterministic rule: it can only name the authors if their names appear
# somewhere in the messages it is sent. The two sections run the same two tasks:
#
# * with memory: one ``Memory`` is shared by both runs, as in ``main.py``;
# * without memory: ``TinyAgent`` always needs a ``Memory``, so "without" means a fresh ``Memory()`` for every run
#   (the phase 2 behaviour, where only the current task was sent).
#
# Expected observations and limits:
# ``NamesLLM`` searches for a fixed name; it is not a realistic language model.
# Assertions inspect both returned text and request contents, including growth
# from one to three to five messages with shared memory. A missing name shows
# absence of supplied context, not proof that a real model cannot guess it.
# ====================================================================================================

# ---- with memory: one Memory shared by both runs --------------------------------------


def test_the_second_run_sends_the_whole_conversation():
    llm = NamesLLM()
    agent = TinyAgent(llm, Memory())
    agent.run(GREETING)
    agent.run(QUESTION)
    assert llm.calls[1] == [
        {"role": "user", "content": GREETING},
        {"role": "assistant", "content": "Nice to meet you!"},
        {"role": "user", "content": QUESTION},
    ]


def test_the_agent_can_answer_from_an_earlier_turn():
    agent = TinyAgent(NamesLLM(), Memory())
    agent.run(GREETING)
    assert agent.run(QUESTION) == "Maarten and Jay."


def test_memory_holds_every_turn_in_order():
    memory = Memory()
    agent = TinyAgent(NamesLLM(), memory)
    agent.run(GREETING)
    agent.run(QUESTION)
    assert [(m["role"], m["content"]) for m in memory.get_messages()] == [
        ("user", GREETING),
        ("assistant", "Nice to meet you!"),
        ("user", QUESTION),
        ("assistant", "Maarten and Jay."),
    ]


def test_the_history_grows_by_two_messages_per_run():
    llm = NamesLLM()
    agent = TinyAgent(llm, Memory())
    for task in (GREETING, QUESTION, QUESTION):
        agent.run(task)
    assert [len(call) for call in llm.calls] == [1, 3, 5]


# ---- without memory: a fresh Memory for every run -------------------------------------


def test_the_second_run_sends_only_the_question():
    llm = NamesLLM()
    TinyAgent(llm, Memory()).run(GREETING)
    TinyAgent(llm, Memory()).run(QUESTION)
    assert llm.calls[1] == [{"role": "user", "content": QUESTION}]


def test_the_agent_cannot_answer_from_an_earlier_run():
    llm = NamesLLM()
    TinyAgent(llm, Memory()).run(GREETING)
    assert TinyAgent(llm, Memory()).run(QUESTION) == "I don't know your names."


def test_each_memory_holds_only_its_own_run():
    first, second = Memory(), Memory()
    TinyAgent(NamesLLM(), first).run(GREETING)
    TinyAgent(NamesLLM(), second).run(QUESTION)
    assert [m["content"] for m in first.get_messages()] == [GREETING, "Nice to meet you!"]
    assert [m["content"] for m in second.get_messages()] == [QUESTION, "I don't know your names."]


def test_the_history_never_grows_across_runs():
    llm = NamesLLM()
    for task in (GREETING, QUESTION, QUESTION):
        TinyAgent(llm, Memory()).run(task)
    assert [len(call) for call in llm.calls] == [1, 1, 1]


# ====================================================================================================
# test_step
# Developer traces: distinguish model replies from agent outcomes.
#
# A response contains what the model returned. A trajectory creates a step
# that records what the harness did with it, so a developer can inspect an
# answer separately from an action and observation.
#
# What this file demonstrates:
# Tests for ``Step`` and ``Trajectory``.
#
# Expected observations and limits:
# The two tests pin empty field defaults and answer recording after a run is
# initialized. They do not test tool-action recording or an entire agent loop;
# compare ``test_tools.py`` for the action branch.
# ====================================================================================================

def test_step_defaults():
    step = Step()
    assert (step.thought, step.action, step.observation, step.answer, step.metadata) == ("", None, None, None, None)


def test_trajectory_records_steps_as_step():
    trajectory = Trajectory()
    trajectory.initialize("q")
    trajectory.add(Response(content="a"))
    (step,) = trajectory.runs[0]["steps"]
    assert type(step) is Step and step.answer == "a"
