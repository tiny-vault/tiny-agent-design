"""What the LLM sends back, as a plain Python object.

A model server replies with nested JSON (``choices[0].message`` and more).
:class:`Response` is the flat shape the rest of the harness works with, so the
agent loop never has to know that format.
"""

from dataclasses import dataclass


@dataclass
class Response:
    """Structured response from LLM calls.

    Built by :meth:`tiny_agent.shared.llm.LLM.generate`. All fields have
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

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None
