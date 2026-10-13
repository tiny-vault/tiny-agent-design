"""Tests for ``TrimmingMemory`` (the sliding window over the conversation) and the flag that turns it on.

Three groups: the memory on its own (what is kept, in which order), the memory inside a ``TinyAgent`` (what
the model is actually sent over several runs), and ``main.memory_from_env`` (the ``TINYAGENT_TRIM_MEMORY``
flag). The agent tests use a stand-in model that records every call, so no server is needed.
"""

from typing import Any

import pytest

import main as demo
from llm_client.llm import LLM
from llm_client.response import Response
from tiny_agent.shared.memory import Memory
from tiny_agent.shared.memory.trimming import TrimmingMemory
from tiny_agent.shared.tiny_agent import TinyAgent


class RecordingLLM(LLM):
    """A stand-in model: answers ``"a<n>"`` for its n-th call and records the messages it was sent."""

    def __init__(self):
        super().__init__(model="recording")
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        # A copy: memory hands out its live list, which changes after the call returns.
        self.calls.append(list(messages))
        return Response(content=f"a{len(self.calls)}")


def texts(memory: Memory) -> list[str]:
    """The message texts in order, which is all these tests need to tell the messages apart."""
    return [message["content"] for message in memory.get_messages()]


def add_turns(memory: Memory, count: int) -> None:
    """Add ``count`` user/assistant pairs: ``u1``, ``a1``, ``u2``, ``a2``, ..."""
    for i in range(1, count + 1):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")


# ---- the memory on its own ----------------------------------------------------------

def test_it_is_a_memory():
    assert isinstance(TrimmingMemory(), Memory)


def test_two_turns_fit_in_the_window_and_nothing_is_dropped():
    memory = TrimmingMemory()
    add_turns(memory, 2)
    assert texts(memory) == ["u1", "a1", "u2", "a2"]


def test_the_third_turn_pushes_the_first_out():
    memory = TrimmingMemory()
    add_turns(memory, 3)
    assert texts(memory) == ["u2", "a2", "u3", "a3"]


def test_the_window_can_start_with_an_assistant_message_right_after_a_user_message():
    """Pins the current behaviour: the trim counts messages, so adding u3 cuts u1 but keeps a1."""
    memory = TrimmingMemory()
    add_turns(memory, 2)
    memory.add("user", "u3")
    assert texts(memory) == ["a1", "u2", "a2", "u3"]


def test_the_history_never_grows_past_four_messages():
    memory = TrimmingMemory()
    for i in range(1, 51):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
        assert len(memory.get_messages()) <= 4
    assert texts(memory) == ["u49", "a49", "u50", "a50"]


def test_plain_memory_keeps_everything_for_contrast():
    memory = Memory()
    add_turns(memory, 3)
    assert len(memory.get_messages()) == 6


def test_the_system_message_is_always_kept_and_stays_first():
    memory = TrimmingMemory()
    memory.add("system", "Be brief.")
    add_turns(memory, 4)
    assert texts(memory) == ["Be brief.", "u3", "a3", "u4", "a4"]


def test_a_system_message_added_late_still_comes_first():
    memory = TrimmingMemory()
    add_turns(memory, 1)
    memory.add("system", "Be brief.")
    assert texts(memory) == ["Be brief.", "u1", "a1"]


def test_every_system_message_is_kept():
    memory = TrimmingMemory()
    memory.add("system", "s1")
    memory.add("system", "s2")
    add_turns(memory, 3)
    assert texts(memory) == ["s1", "s2", "u2", "a2", "u3", "a3"]


def test_a_tool_call_is_stored_like_in_memory_and_a_tool_message_counts_toward_the_window():
    call = {"id": "1", "function": {"name": "f", "arguments": "{}"}}
    memory = TrimmingMemory()
    memory.add("user", "u1")
    memory.add("assistant", "", tool_call=call)
    memory.add("tool", "result")
    memory.add("assistant", "a1")
    assert memory.get_messages()[1] == {"role": "assistant", "content": "", "tool_calls": [call]}
    memory.add("user", "u2")
    # Five non-system messages: the oldest (u1) is out, the tool message is one of the four kept.
    assert [m["role"] for m in memory.get_messages()] == ["assistant", "tool", "assistant", "user"]


def test_get_messages_returns_the_trimmed_history_itself():
    memory = TrimmingMemory()
    add_turns(memory, 3)
    assert memory.get_messages() is memory.messages


# ---- the memory inside a TinyAgent ----------------------------------------------------------

def test_the_model_is_sent_only_the_window_and_forgets_the_first_task():
    llm = RecordingLLM()
    agent = TinyAgent(llm, TrimmingMemory())
    for task in ("u1", "u2", "u3", "u4"):
        agent.run(task)
    # How many messages each call carried: the window stops growing at four.
    assert [len(call) for call in llm.calls] == [1, 3, 4, 4]
    # By the fourth call the first task is gone from what the model sees.
    assert "u1" not in [message["content"] for message in llm.calls[3]]
    assert [message["content"] for message in llm.calls[3]] == ["a2", "u3", "a3", "u4"]


def test_with_plain_memory_the_same_runs_keep_the_first_task():
    llm = RecordingLLM()
    agent = TinyAgent(llm, Memory())
    for task in ("u1", "u2", "u3", "u4"):
        agent.run(task)
    assert [len(call) for call in llm.calls] == [1, 3, 5, 7]
    assert llm.calls[3][0]["content"] == "u1"


# ---- the flag ----------------------------------------------------------------------------

def test_the_flag_is_off_by_default():
    assert type(demo.memory_from_env()) is Memory


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_these_values_turn_trimming_on(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", value)
    assert type(demo.memory_from_env()) is TrimmingMemory


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "maybe"])
def test_other_values_leave_trimming_off(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", value)
    assert type(demo.memory_from_env()) is Memory


def test_the_demo_gives_the_same_answer_with_the_flag_on(demo_server, monkeypatch, capsys):
    """The demo's two tasks fit in the window, so trimming changes nothing it sends or prints."""
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    demo.main()
    assert capsys.readouterr().out.strip() == "Your names are Maarten and Jay."
    assert [m["content"] for m in demo_server.requests[1]["body"]["messages"]][0] == demo.GREETING
