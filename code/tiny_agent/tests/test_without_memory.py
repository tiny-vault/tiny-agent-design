"""Checks that a ``TinyAgent`` WITHOUT memory forgets: each run starts from nothing.

``TinyAgent`` always needs a ``Memory``, so "without memory" here means a fresh ``Memory()`` for every run (the phase 2 behaviour, where only the current task was sent). ``NamesLLM`` answers the way a real model would: it can only name the authors if their names appear somewhere in the messages it is sent. Compare ``test_with_memory.py``, which shares one ``Memory`` across both runs.
"""

from llm_client.llm import LLM
from llm_client.response import Response
from tiny_agent.shared.memory import Memory
from tiny_agent.shared.tiny_agent import TinyAgent

GREETING = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."
QUESTION = "Hi! What are our names?"


class NamesLLM(LLM):
    """A model that only knows what it is sent: it names the authors if "Maarten" is in the messages, else it admits it does not know."""

    def __init__(self):
        super().__init__(model="names")
        self.calls = []

    def generate(self, messages: list[dict], tools: list | None = None) -> Response:
        # A copy: memory hands out its live list, which grows after the call returns.
        self.calls.append(list(messages))
        if messages[-1]["content"] == QUESTION:
            seen = any("Maarten" in message["content"] for message in messages[:-1])
            return Response(content="Maarten and Jay." if seen else "I don't know your names.")
        return Response(content="Nice to meet you!")


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
