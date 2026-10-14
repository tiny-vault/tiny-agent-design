"""What the LLM sends back, as a plain Python object.

A model server replies with nested JSON (``choices[0].message`` and more).
:class:`Response` is the flat shape the rest of the harness works with, so the
agent loop never has to know that format.

Why it exists: different servers spell the same thing differently (llama.cpp
sends the model's thinking as ``reasoning_content``, Ollama as ``reasoning``).
Converting once, in :meth:`tiny_agent.components.client.llm.LLM.generate`, keeps that mess out of
the agent. The key idea for agents is that one reply is either a final answer
(``content``) or a request to run a tool (``tool_call``); the loop branches on
which one it got.

This file defines the data container and makes no external calls or
server-response parsing; pydantic validates field values at construction.
Text and a tool call can coexist, so a tool-enabled agent branches on the
requested action rather than treating content as proof of completion.

The initial tool call has the server's nested ``function`` shape. A registry
can return a new response with ``{"tool": name, "kwargs": arguments}``
instead. This field accepts either dict shape but does not validate the tool's
argument schema. Compare ``Step`` for the record that adds the outcome.
"""

from typing import Any

from pydantic import BaseModel


class Response(BaseModel):
    """Structured response from LLM calls.

    Built by :meth:`tiny_agent.components.client.llm.LLM.generate`. All fields have
    defaults, so ``Response()`` is a valid empty response.

    A reply is usually one of two things: a final answer (``content`` is set,
    ``tool_call`` is ``None``) or a request to run a tool (``tool_call`` is set
    and ``content`` is often ``None``). The agent loop tells them apart by
    checking ``tool_call``.

    Attributes:
        content: The answer text. ``None`` when the model replied only with a
            tool call (the default is an empty string).
        reasoning: The model's thinking, when it sends it separately and thinking
            is on. ``None`` otherwise.
        tool_call: The tool the model wants run, or ``None``. It keeps the
            server's shape::

                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": '{"city": "Paris"}'}}

            Note that ``arguments`` is a JSON *string*; decode it with
            ``json.loads`` before use. Only the first call is kept when the model
            asks for several.
        metadata: Bookkeeping about the call: ``model`` (which model answered),
            ``prompt_tokens`` and ``completion_tokens``. ``None`` if not set.
    """

    content: str | None = ""
    reasoning: str | None = None
    tool_call: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
