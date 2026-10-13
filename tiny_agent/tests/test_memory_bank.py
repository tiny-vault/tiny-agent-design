"""Tests for ``MemoryBank`` (event summaries, a user portrait and a forgetting curve).

Stand-ins replace both models so the tests are fast and exact: ``FakeLLM`` answers ``"summary<n>"`` or
``"portrait<n>"`` depending on the prompt and records every call, ``FakeEmbedder`` maps a text to a vector by the
key words it contains, and ``Clock`` is a time the tests move by hand.
"""

from typing import Any

from tiny_agent.agent.memory import Memory
from tiny_agent.agent.tiny_agent import TinyAgent
from tiny_agent.llm_client.llm import LLM
from tiny_agent.llm_client.response import Response
from tiny_agent.memory.memory_bank import (
    FORGET_BELOW,
    SECONDS_PER_DAY,
    TOP_K,
    MemoryBank,
)

DAY = SECONDS_PER_DAY


class FakeLLM(LLM):
    """Answers ``summary<n>`` to a summary prompt and ``portrait<n>`` to a portrait prompt; records the prompts."""

    def __init__(self):
        super().__init__(model="fake")
        self.prompts: list[str] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        kind = "portrait" if prompt.startswith("Update the user's portrait") else "summary"
        return Response(content=f"{kind}{len(self.prompts)}")


class FakeEmbedder:
    """Vector of key-word counts: ``cat``, ``dog``, ``fish``. Counts its requests."""

    WORDS = ("cat", "dog", "fish")

    def __init__(self):
        self.requests: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.requests.append(text)
        return [float(text.lower().count(word)) + 0.01 for word in self.WORDS]


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make() -> tuple[MemoryBank, FakeLLM, FakeEmbedder, Clock]:
    llm, embedder, clock = FakeLLM(), FakeEmbedder(), Clock()
    return MemoryBank(llm, embedder, clock), llm, embedder, clock  # pyright: ignore[reportArgumentType]


def bank_with(texts: list[str]) -> tuple[MemoryBank, Clock]:
    """A bank that already holds one event summary per text (put there directly, no model calls)."""
    memory, _, embedder, clock = make()
    memory.event_summaries = [
        {"text": t, "embedding": embedder.embed(t), "strength": 1, "last_recalled": clock.now} for t in texts
    ]
    return memory, clock


# ---- storage ----------------------------------------------------------------------------

def test_it_is_a_memory_and_starts_empty():
    memory, llm, embedder, _ = make()
    assert isinstance(memory, Memory)
    assert memory.get_messages() == []
    assert memory.conversation_history == [] and memory.event_summaries == [] and memory.user_portrait == ""
    assert llm.prompts == [] and embedder.requests == []


def test_every_message_is_kept_raw_with_its_time():
    memory, _, _, clock = make()
    memory.add("user", "I have a cat")
    clock.now += 5
    memory.add("assistant", "Nice")
    assert memory.conversation_history == [
        {"role": "user", "content": "I have a cat", "time": 1_000_000.0},
        {"role": "assistant", "content": "Nice", "time": 1_000_005.0},
    ]


def test_an_assistant_reply_stores_an_event_summary_and_a_portrait():
    memory, llm, _, _ = make()
    memory.add("user", "I have a cat")
    memory.add("assistant", "Nice")
    assert [s["text"] for s in memory.event_summaries] == ["summary1"]
    assert memory.event_summaries[0]["strength"] == 1
    assert memory.user_portrait == "portrait2"
    assert len(llm.prompts) == 2
    assert "user: I have a cat\nassistant: Nice\n" in llm.prompts[0]


def test_a_second_turn_summarises_only_the_new_turn_and_extends_the_portrait():
    memory, llm, _, _ = make()
    for i in (1, 2):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
    assert "u1" not in llm.prompts[2] and "user: u2\nassistant: a2\n" in llm.prompts[2]
    assert "Portrait: portrait2\n" in llm.prompts[3]
    assert [s["text"] for s in memory.event_summaries] == ["summary1", "summary3"]


def test_user_and_tool_messages_alone_make_no_model_call():
    memory, llm, _, _ = make()
    memory.add("user", "u1")
    memory.add("tool", "result")
    assert llm.prompts == []


# ---- retrieval ---------------------------------------------------------------------------

def test_search_returns_the_most_similar_summaries_first_and_at_most_top_k():
    memory, _ = bank_with(["cat cat", "dog dog", "fish fish", "cat dog", "cat"])
    found = memory.search("my cat")
    assert len(found) == TOP_K and found[0] in ("cat cat", "cat")


def test_search_with_nothing_stored_makes_no_embedding_request():
    memory, _, embedder, _ = make()
    assert memory.search("cat") == []
    assert embedder.requests == []


def test_a_user_message_gets_the_portrait_and_memories_in_front_of_it():
    memory, _ = bank_with(["dog walk"])
    memory.user_portrait = "calm"
    memory.add("user", "my dog?")
    assert memory.get_messages()[0]["content"] == "Portrait:\ncalm\n\nMemories:\ndog walk\n\nQuestion: my dog?"
    # The raw history keeps what was said, not the augmented text.
    assert memory.conversation_history[0]["content"] == "my dog?"


def test_other_roles_are_stored_unchanged():
    memory, _ = bank_with(["dog walk"])
    memory.add("tool", "result")
    assert memory.get_messages() == [{"role": "tool", "content": "result"}]


# ---- forgetting and reinforcement ----------------------------------------------------------

def test_recalling_a_summary_adds_one_to_its_strength_and_restarts_its_clock():
    memory, clock = bank_with(["dog walk"])
    clock.now += 2 * DAY
    memory.search("dog")
    item = memory.event_summaries[0]
    assert item["strength"] == 2 and item["last_recalled"] == clock.now


def test_a_summary_not_recalled_for_long_enough_is_forgotten_but_kept():
    memory, clock = bank_with(["dog walk"])
    clock.now += 3 * DAY  # exp(-3) is about 0.05, below FORGET_BELOW
    assert memory.search("dog") == []
    assert len(memory.event_summaries) == 1 and FORGET_BELOW > 0.05


def test_a_stronger_summary_outlasts_a_weaker_one():
    memory, clock = bank_with(["dog walk", "dog park"])
    memory.event_summaries[1]["strength"] = 5
    clock.now += 3 * DAY  # exp(-3/5) is about 0.55
    assert memory.search("dog") == ["dog park"]


def test_an_unrecalled_summary_stays_until_its_retention_drops_below_the_limit():
    memory, clock = bank_with(["dog walk"])
    clock.now += 2 * DAY  # exp(-2) is about 0.14, still above
    assert memory.search("dog") == ["dog walk"]


# ---- inside a TinyAgent --------------------------------------------------------------------

def test_the_agent_gets_an_answer_call_then_two_memory_calls_per_task():
    memory, _, _, _ = make()
    answers = []

    class Answering(FakeLLM):
        def generate(self, messages, tools=None):
            if messages[0]["content"].startswith(("Summarize", "Update the user's portrait")):
                return super().generate(messages)
            answers.append(list(messages))
            return Response(content="ok")

    answering = Answering()
    memory.llm = answering
    agent = TinyAgent(answering, memory)
    assert agent.run("I have a cat") == "ok"
    assert agent.run("what pet do I have?") == "ok"
    # The second question reached the model with the first turn's summary in front of it.
    assert "Memories:\nsummary1" in answers[1][-1]["content"]
