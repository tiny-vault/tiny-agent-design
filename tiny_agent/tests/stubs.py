"""Test doubles: make model responses and request histories deterministic.

A double replaces model generation with a small rule, letting a test isolate
harness behavior from stochastic replies. Reusing a client does not give it
conversation memory; it records the history its caller supplies.

What this file demonstrates:
Stand-ins and small helpers that several test files share, so each is written once.

The stub models are subclasses of ``LLM`` so they type-check where ``TinyAgent`` or a memory expects an ``LLM``;
none of them touches the network. Each records what it was sent:

* ``calls``: the messages of every call, oldest first. They are copies, because memory hands out its live list,
  which grows after the call returns.
* ``tools``: the ``tools`` argument of every call, in the same order.

Import from here, for example ``from tiny_agent.tests.stubs import ScriptedLLM``.

Expected observations and limits:
The message lists are shallow copies: appending later messages does not alter
a recorded call, but mutating an existing message dict can. ``ScriptedLLM``
raises ``IndexError`` when replies run out. ``NamesLLM`` recognizes a fixed
name rather than performing language understanding. ``write_env`` writes a
temporary settings file and changes the selected path for the caller.
"""

import os
from pathlib import Path
from typing import Any

from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response

# Where the real models of ``node execute.mjs up llm-client`` answer (the tests that need them skip when they do not).
EMBEDDING_URL = os.environ.get("TINYAGENT_EMBEDDING_URL", "http://127.0.0.1:3201/v1")
LLM_URL = os.environ.get("TINYAGENT_LLM_URL", "http://127.0.0.1:3200/v1")

GREETING = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."
QUESTION = "Hi! What are our names?"


class StubLLM(LLM):
    """A model that records every call and answers with :meth:`reply`; a subclass decides the answer."""

    def __init__(self, model: str = "stub"):
        super().__init__(model=model)
        self.calls: list[list[dict[str, Any]]] = []
        self.tools: list[list[Any] | None] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        """Record shallow message copies and tool definitions, then wrap ``reply``.

        This replaces HTTP generation; subclasses return text or a ready Response.
        Script exhaustion and errors from ``reply`` propagate to the test.
        """
        self.calls.append(list(messages))
        self.tools.append(tools)
        reply = self.reply(messages)
        return reply if isinstance(reply, Response) else Response(content=reply)

    def reply(self, messages: list[dict[str, Any]]) -> str | Response:
        """The answer to one call; ``len(self.calls)`` is the number of this call (it is already counted)."""
        raise NotImplementedError


class EchoLLM(StubLLM):
    """Answers ``"Received: <last message>"``, so a test can assert exactly what the agent sent and returned."""

    def __init__(self):
        super().__init__(model="echo")

    def reply(self, messages: list[dict[str, Any]]) -> str:
        return f"Received: {messages[-1]['content']}"


class ScriptedLLM(StubLLM):
    """Answers with the given replies (text or ``Response``) in order."""

    def __init__(self, *replies: str | Response):
        super().__init__(model="scripted")
        self.replies = list(replies)

    def reply(self, messages: list[dict[str, Any]]) -> str | Response:
        return self.replies.pop(0)


class RecordingLLM(StubLLM):
    """Answers ``"<prefix><n>"`` for its n-th call (``s1``, ``s2``, ...), or always ``content`` when that is given."""

    def __init__(self, content: str | None = None, prefix: str = "s"):
        super().__init__(model="recording")
        self.fixed = content
        self.prefix = prefix

    def reply(self, messages: list[dict[str, Any]]) -> str:
        return self.fixed if self.fixed is not None else f"{self.prefix}{len(self.calls)}"


class NamesLLM(StubLLM):
    """A model that only knows what it is sent: it names the authors if "Maarten" is in the messages, else it admits it does not know."""

    def __init__(self):
        super().__init__(model="names")

    def reply(self, messages: list[dict[str, Any]]) -> str:
        if messages[-1]["content"] == QUESTION:
            seen = any("Maarten" in message["content"] for message in messages[:-1])
            return "Maarten and Jay." if seen else "I don't know your names."
        return "Nice to meet you!"


def texts(memory: Any) -> list[str]:
    """The message texts of a memory in order, which is all most memory tests need to tell the messages apart."""
    return [message["content"] for message in memory.get_messages()]


def llm(fake_server: Any, model: str = "a", **kwargs: Any) -> LLM:
    """A real ``LLM`` pointed at the fake server (for the tests of the client and the CLI)."""
    return LLM(model=model, base_url=fake_server.url, **kwargs)


def write_env(tmp_path: Path, text: str) -> Path:
    """Write a settings file and make the code read it instead of the real ``.env``."""
    path = tmp_path / "settings.env"
    path.write_text(text, encoding="utf-8")
    os.environ["TINYAGENT_ENV_FILE"] = str(path)
    return path
