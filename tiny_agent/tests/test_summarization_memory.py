"""Tests for ``SummarizationMemory`` (a running summary written by the model) and the flag that turns it on.

Three groups: the memory on its own (when it summarises, what it asks the model, what it keeps), the memory
inside a ``TinyAgent`` (what the model is actually sent over several runs), and ``main.memory_from_env`` (the
``TINYAGENT_SUMMARY_MEMORY`` flag). A stand-in model records every call and answers ``"s<n>"``, so no server is
needed except in the last test.
"""

from typing import Any

import pytest

from tiny_agent import main as demo
from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import BaseMemory, Memory
from tiny_agent.components.memory.summarization import SummarizationMemory
from tiny_agent.components.memory.trimming import TrimmingMemory


class RecordingLLM(LLM):
    """A stand-in model: answers ``"s<n>"`` for its n-th call and records the messages it was sent."""

    def __init__(self, content: str | None = None):
        super().__init__(model="recording")
        self.fixed = content
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        # A copy: memory hands out its live list, which changes after the call returns.
        self.calls.append(list(messages))
        return Response(content=self.fixed if self.fixed is not None else f"s{len(self.calls)}")


def texts(memory: BaseMemory) -> list[str]:
    """The message texts in order, which is all these tests need to tell the messages apart."""
    return [message["content"] for message in memory.get_messages()]


# ---- the memory on its own ----------------------------------------------------------

def test_it_is_a_memory():
    assert isinstance(SummarizationMemory(RecordingLLM()), BaseMemory)


def test_it_starts_empty_and_makes_no_model_call():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    assert memory.get_messages() == []
    assert llm.calls == []


def test_user_and_tool_messages_are_stored_without_a_model_call():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("tool", "result")
    assert texts(memory) == ["u1", "result"]
    assert llm.calls == []


def test_an_assistant_reply_replaces_the_history_with_one_system_summary():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    assert memory.get_messages() == [{"role": "system", "content": "s1"}]
    assert len(llm.calls) == 1


def test_the_summary_prompt_holds_the_old_summary_and_the_new_turn():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    memory.add("user", "u2")
    memory.add("assistant", "a2")
    first, second = (call[0]["content"] for call in llm.calls)
    assert [len(call) for call in llm.calls] == [1, 1]
    assert "Summary: \n" in first and "user: u1\nassistant: a1\n" in first
    # The second prompt carries the first summary ("s1") and only the turn added since.
    assert "Summary: s1\n" in second and "user: u2\nassistant: a2\n" in second
    assert "u1" not in second
    assert texts(memory) == ["s2"]


def test_a_user_message_after_a_summary_is_kept_next_to_it():
    memory = SummarizationMemory(RecordingLLM())
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    memory.add("user", "u2")
    assert texts(memory) == ["s1", "u2"]


def test_the_history_never_grows_past_one_message_after_a_reply():
    memory = SummarizationMemory(RecordingLLM())
    for i in range(1, 21):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
        assert len(memory.get_messages()) == 1
    assert texts(memory) == ["s20"]


def test_a_reply_with_no_text_keeps_the_old_summary():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    llm.fixed = ""
    memory.add("user", "u2")
    memory.add("assistant", "a2")
    assert texts(memory) == ["s1"]


def test_a_tool_call_is_stored_like_in_memory():
    call = {"id": "1", "function": {"name": "f", "arguments": "{}"}}
    memory = SummarizationMemory(RecordingLLM())
    memory.add("user", "u1")
    memory.add("tool", "result", tool_call=call)
    assert memory.get_messages()[1] == {"role": "tool", "content": "result", "tool_calls": [call]}


def test_get_messages_returns_the_history_itself():
    memory = SummarizationMemory(RecordingLLM())
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    assert memory.get_messages() is memory.messages


def test_an_existing_system_message_is_taken_for_the_summary_and_overwritten():
    """Pins the current behaviour: the summary lives in the system message, so a system prompt does not survive."""
    memory = SummarizationMemory(RecordingLLM())
    memory.add("system", "Be brief.")
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    assert texts(memory) == ["s1"]


def test_a_failing_model_raises_and_leaves_the_reply_unsummarised():
    class Broken(RecordingLLM):
        def generate(self, messages, tools=None):
            raise RuntimeError("server down")

    memory = SummarizationMemory(Broken())
    memory.add("user", "u1")
    with pytest.raises(RuntimeError, match="server down"):
        memory.add("assistant", "a1")
    assert texts(memory) == ["u1", "a1"]


# ---- the memory inside a TinyAgent ----------------------------------------------------------

def test_the_model_is_sent_the_summary_and_the_new_question_each_run():
    llm = RecordingLLM()
    agent = TinyAgent(llm, SummarizationMemory(llm))
    for task in ("u1", "u2", "u3"):
        agent.run(task)
    # Calls alternate: the agent's answer call, then the memory's summary call (one message, the prompt).
    answer_calls = llm.calls[0::2]
    summary_calls = llm.calls[1::2]
    assert [[m["content"] for m in call] for call in answer_calls] == [
        ["u1"],
        ["s2", "u2"],  # s2 is the 2nd model call overall: the summary of the first turn
        ["s4", "u3"],
    ]
    assert all(len(call) == 1 and call[0]["role"] == "user" for call in summary_calls)
    # Three answers and three summaries.
    assert len(llm.calls) == 6


def test_the_first_task_is_not_resent_after_it_is_summarised():
    llm = RecordingLLM()
    agent = TinyAgent(llm, SummarizationMemory(llm))
    for task in ("u1", "u2", "u3"):
        agent.run(task)
    assert "u1" not in [m["content"] for m in llm.calls[4]]


# ---- the flag ----------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_these_values_turn_summarizing_on(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", value)
    assert type(demo.memory_from_env(RecordingLLM())) is SummarizationMemory


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_other_values_leave_summarizing_off(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", value)
    assert type(demo.memory_from_env(RecordingLLM())) is Memory


def test_the_memory_gets_the_llm_it_was_given(monkeypatch):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    llm = RecordingLLM()
    memory = demo.memory_from_env(llm)
    assert isinstance(memory, SummarizationMemory) and memory.llm is llm


def test_summarizing_needs_an_llm(monkeypatch):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    with pytest.raises(ValueError, match="llm"):
        demo.memory_from_env()


def test_the_summary_flag_wins_over_the_trim_flag(monkeypatch):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    assert type(demo.memory_from_env(RecordingLLM())) is SummarizationMemory


def test_the_trim_flag_still_works_when_the_summary_flag_is_off(monkeypatch):
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    assert type(demo.memory_from_env(RecordingLLM())) is TrimmingMemory


def test_the_demo_sends_the_summary_instead_of_the_history(demo_server, monkeypatch, capsys):
    """With the flag on, the second task carries the summary, not the greeting: four requests in all."""
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    demo.main()
    assert capsys.readouterr().out.strip() == "Your names are Maarten and Jay."
    assert len(demo_server.requests) == 4  # greeting, summary, question, summary
    second_task = demo_server.requests[2]["body"]["messages"]
    assert [m["role"] for m in second_task] == ["system", "user"]
    assert second_task[1]["content"] == demo.QUESTION
    assert demo.GREETING not in second_task[0]["content"]
