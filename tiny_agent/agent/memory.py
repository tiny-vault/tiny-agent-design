"""Memory: the conversation history the agent sends to the model.

A model server is stateless: it forgets everything between requests. An agent
that should remember what was said (or what a tool returned) therefore has to
keep the history itself and send all of it with every request. That stored list
of messages is the agent's short-term memory.

Each message is a dict in the OpenAI chat format, ``{"role": ..., "content": ...}``,
where the role is ``system``, ``user``, ``assistant`` or ``tool``. A message that
records a tool call also carries ``tool_calls``.

This file keeps the list in plain Python memory only. It is lost when the program
ends, never shortened, and has no limit, so a long chat will eventually exceed the
model's context window. ``trimming.TrimmingMemory`` (in the ``tiny_agent.memory`` package) is the first
fix: it keeps only the last two turns. ``summarization.SummarizationMemory`` keeps a running summary
instead. Saving to disk is left to a later phase.
"""

from typing import Any


class Memory:
    """Simple memory module to store conversation history.

    Role: the "context" slot of the harness. The agent adds each user message and
    each model reply here, and reads the whole list back before every model call.

    Attributes:
        messages: The history so far, oldest first, in chat-message format.
    """

    def __init__(self):
        """Start with an empty history."""
        self.messages = []

    def add(
        self, role: str, content: str, tool_call: dict[str, Any] | None = None, **kwargs
    ) -> None:
        """Add a message to memory.

        Args:
            role: Who said it: ``system``, ``user``, ``assistant`` or ``tool``.
            content: The message text.
            tool_call: The tool call the model made, if this message records one.
                It is stored under ``tool_calls`` as a one-item list, the shape
                chat servers expect.
            **kwargs: Accepted and ignored, so callers may pass extra details.
        """
        message: dict[str, Any] = {"role": role, "content": content}

        # Tool call: the server expects a list, even for a single call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # Append message to memory
        self.messages.append(message)

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages.

        Returns:
            The history itself, not a copy, so changing it changes the memory. This
            is what the agent passes to ``LLM.generate`` on each step.
        """
        return self.messages
