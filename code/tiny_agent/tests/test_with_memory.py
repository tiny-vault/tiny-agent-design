"""Checks that a ``TinyAgent`` WITH memory carries earlier turns into later ones.

One ``Memory`` is shared by both runs, as in ``main.py``. ``NamesLLM`` answers the way a real model would: it can only name the authors if their names appear somewhere in the messages it is sent. Compare ``test_without_memory.py``, which runs the same two tasks with a fresh ``Memory`` each time.
"""

from typing import Any

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

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        # A copy: memory hands out its live list, which grows after the call returns.
        self.calls.append(list(messages))
        if messages[-1]["content"] == QUESTION:
            seen = any("Maarten" in message["content"] for message in messages[:-1])
            return Response(content="Maarten and Jay." if seen else "I don't know your names.")
        return Response(content="Nice to meet you!")


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
