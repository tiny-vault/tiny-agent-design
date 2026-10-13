"""MemoryBank: long-term memory that fades and strengthens, kept outside the prompt (Zhong et al., 2023).

WHAT IS IT? ``Memory`` is short-term: the conversation, resent on every call. ``RAGMemory`` retrieves from documents
someone wrote in advance. ``MemoryBank`` builds its own long-term store from the conversations themselves and keeps
it up to date. It holds three things:

* ``conversation_history``: every message, raw and timestamped.
* ``event_summaries``: one short summary per completed turn, written by the model, each with a vector, a
  strength and the time it was last recalled.
* ``user_portrait``: the user's personality traits and emotions, rewritten by the model after each turn.

HOW IT WORKS: when a ``user`` message is added, it is embedded and ``search`` ranks the event summaries by cosine
similarity. The best ``TOP_K`` are put in front of the message together with the portrait, as ``Portrait:``,
``Memories:`` and ``Question:``. Each recalled summary gets stronger. When the ``assistant`` reply is added, the
turn is summarised (one model call, one embedding) and the portrait is updated (a second model call).
Forgetting follows the Ebbinghaus curve, ``retention = exp(-t / S)``: ``t`` is the days since the summary was last
recalled and ``S`` its strength. A new summary has ``S = 1``; every recall adds 1 to ``S`` and restarts ``t``, so
what is used often lasts longer.

WHAT TO NOTICE: the paper calls its updater "exploratory and highly simplified" and gives no deletion rule; the
choices here are ours. A summary whose retention is below ``FORGET_BELOW`` is skipped by ``search`` but kept in
the list, and similarity alone ranks the rest. ``messages`` is still the plain, ever-growing conversation of the
running program (retrieved context included, as in ``RAGMemory``); the bank is what a later session could
retrieve from, but it lives in Python memory only and is lost when the program ends. Every completed turn costs
two chat calls and one embedding request, made by the memory itself and not recorded in the trajectory; a failing
server raises to the caller. The paper also builds a summary per day and a global one; here it is per turn.
The time comes from ``clock`` (seconds, ``time.time`` by default) so tests can move it.
"""

import math
import time
from collections.abc import Callable
from typing import Any

from tiny_agent.agent.memory import Memory
from tiny_agent.llm_client.embedding import EmbeddingModel
from tiny_agent.llm_client.llm import LLM

TOP_K = 3
FORGET_BELOW = 0.1
SECONDS_PER_DAY = 86400


class MemoryBank(Memory):
    """Long-term memory with event summaries, a user portrait and a forgetting curve.

    Role: a drop-in replacement for ``Memory`` in the harness's "context" slot, for agents that should remember
    what past conversations were about. It has the same methods; only ``add`` differs, and it needs a chat
    model and an embedding model.

    Attributes:
        messages: The conversation so far, as in ``Memory``; user messages include their retrieved context.
        llm: The model that writes the event summaries and the portrait.
        embedding_model: The model that turns summaries and questions into vectors.
        clock: Returns the current time in seconds.
        conversation_history: Every message as ``{"role", "content", "time"}``, oldest first.
        event_summaries: One dict per summarised turn: ``text``, ``embedding``, ``strength`` and ``last_recalled``.
        user_portrait: The model's current description of the user; empty until the first reply.
    """

    def __init__(self, llm: LLM, embedding_model: EmbeddingModel, clock: Callable[[], float] = time.time):
        """Start with an empty history, no summaries and no portrait.

        Args:
            llm: The model asked to summarise each turn and update the portrait.
            embedding_model: The model that embeds each summary and each question.
            clock: Returns the current time in seconds; replace it to test forgetting.
        """
        super().__init__()
        self.llm = llm
        self.embedding_model = embedding_model
        self.clock = clock
        self.conversation_history: list[dict[str, Any]] = []
        self.event_summaries: list[dict[str, Any]] = []
        self.user_portrait = ""
        self._turn_start = 0  # index in conversation_history of the first message not yet summarised

    def add(self, role: str, content: str, tool_call: dict[str, Any] | None = None, **kwargs) -> None:
        """Add a message; a user message gets memories in front of it, an assistant reply updates the bank.

        Args:
            role: Who said it: ``system``, ``user``, ``assistant`` or ``tool``.
            content: The message text. For ``user`` it becomes ``Portrait:``, ``Memories:`` and ``Question:``
                plus this text (the original text is what ``conversation_history`` keeps).
            tool_call: The tool call the model made, if this message records one (stored as in ``Memory``).
            **kwargs: Accepted and ignored, as in ``Memory``.
        """
        self.conversation_history.append({"role": role, "content": content, "time": self.clock()})

        # Augment user queries with the portrait and the memories they recall before storing
        stored = content
        if role == "user":
            memories = "\n".join(self.search(content))
            stored = f"""Portrait:
{self.user_portrait}

Memories:
{memories}

Question: {content}"""
        super().add(role, stored, tool_call, **kwargs)

        # A completed turn becomes a new memory and updates the portrait
        if role == "assistant":
            self._remember_turn()

    def search(self, query: str) -> list[str]:
        """Return the event summaries most similar to the query that have not faded, and strengthen them.

        Args:
            query: The text to look up (it is embedded with one request, unless there is nothing to search).

        Returns:
            Up to ``TOP_K`` summary texts, most similar first.
        """
        now = self.clock()
        remembered = [item for item in self.event_summaries if self._retention(item, now) >= FORGET_BELOW]
        if not remembered:
            return []
        query_embed = self.embedding_model.embed(query)
        ranked = sorted(remembered, key=lambda item: self._cosine(query_embed, item["embedding"]), reverse=True)
        top = ranked[:TOP_K]
        # Recalling a memory strengthens it and restarts its clock
        for item in top:
            item["strength"] += 1
            item["last_recalled"] = now
        return [item["text"] for item in top]

    def _remember_turn(self) -> None:
        """Summarise the turn that just finished into an event summary and update the portrait."""
        turn = self.conversation_history[self._turn_start:]
        self._turn_start = len(self.conversation_history)
        conversation = "".join(f"{m['role']}: {m['content']}\n" for m in turn)

        # Summarise the events of the turn and keep the summary with a vector, strength 1 and the time
        summary = self._ask(
            f"""Summarize the events and key information in the content.

Content:
{conversation}
Output the summary only."""
        )
        if summary:
            self.event_summaries.append(
                {
                    "text": summary,
                    "embedding": self.embedding_model.embed(summary),
                    "strength": 1,
                    "last_recalled": self.clock(),
                }
            )

        # Extend the portrait with what the turn shows about the user
        self.user_portrait = (
            self._ask(
                f"""Update the user's portrait: their personality traits and emotions.

Portrait: {self.user_portrait}

Conversation:
{conversation}
Output the updated portrait only."""
            )
            or self.user_portrait
        )

    def _ask(self, prompt: str) -> str:
        """Send one prompt to the model and return its text ("" if the reply has none).

        Args:
            prompt: The instruction, with everything the model needs in it.

        Returns:
            The reply text.
        """
        response = self.llm.generate([{"role": "user", "content": prompt}])
        return response.content or ""

    def _retention(self, item: dict[str, Any], now: float) -> float:
        """Ebbinghaus retention of a summary: ``exp(-t / S)``.

        Args:
            item: An entry of ``event_summaries``.
            now: The current time in seconds.

        Returns:
            1 right after the summary was written or recalled, falling towards 0 as days pass; slower when ``strength`` is higher.
        """
        days = (now - item["last_recalled"]) / SECONDS_PER_DAY
        return math.exp(-days / item["strength"])

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            a: One embedding.
            b: Another embedding of the same length.

        Returns:
            1 for the same direction, near 0 for unrelated texts.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm
