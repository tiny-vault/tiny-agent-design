"""Summarization memory: replace the conversation with a running summary written by the model.

WHAT IS IT? ``Memory`` keeps every message and ``TrimmingMemory`` forgets everything older than two turns.
``SummarizationMemory`` sits between them: after each reply it asks the model to fold the new turn into a short
summary and keeps only that summary, as a single ``system`` message. The history never grows, yet the facts the
summary captured (the user's name, a decision) survive far longer than in a sliding window.

HOW IT WORKS: ``add`` stores the new message with the parent class. When that message is the assistant's reply
(the turn is complete) it builds a prompt from the current summary plus the messages added since, sends it to
the ``llm`` it was given, and replaces ``messages`` with one ``system`` message holding the model's updated
summary. The next user message is appended after it, so the model is sent ``[summary, user]``.

WHAT TO NOTICE: every completed turn costs one extra model call, made by the memory itself, so a run is slower
and the summary call is not recorded in the agent's trajectory. The summary is only as good as the model that
writes it: whatever it leaves out is gone. Because the summary lives in the ``system`` message, a system prompt
added to this memory is treated as the summary and overwritten after the first reply; keep standing
instructions in the summary prompt or in a different slot. The call is not wrapped in a try: if the server
fails, the error reaches the caller and the reply stays in ``messages`` unsummarised.

Switch it on in the demo with ``TINYAGENT_SUMMARY_MEMORY=1`` (see ``main.memory_from_env``); the harness
itself takes whichever memory it is given.
"""

from typing import Any, override

from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import BaseMemory


class SummarizationMemory(BaseMemory):
    """Memory that compresses past turns into a running summary via an LLM.

    Role: a drop-in replacement for ``Memory`` in the harness's "context" slot, for chats that should stay
    short without forgetting everything older than a few turns. It has the same methods; only ``add``
    differs, and it needs a model to write the summaries.

    Attributes:
        messages: Empty at the start; after the first reply, one ``system`` message holding the summary, and
            after that the summary plus whatever was added since the last reply.
        llm: The model that writes the summary (it may be the same one the agent uses).
    """

    def __init__(self, llm: LLM):
        """Start with an empty history.

        Args:
            llm: The model asked to update the summary after each reply.
        """
        super().__init__()
        self.llm = llm

    @override
    def add(self, role: str, content: str, tool_call: dict[str, Any] | None = None, **kwargs) -> None:
        """Add a message; after an assistant reply, replace the history with an updated summary.

        Args:
            role: Who said it: ``system``, ``user``, ``assistant`` or ``tool``.
            content: The message text.
            tool_call: The tool call the model made, if this message records one (stored as in ``Memory``).
            **kwargs: Accepted and ignored, as in ``Memory``.
        """
        # Add the new message first using the parent class's helper (BaseMemory)
        self._append(role, content, tool_call)

        # After each completed turn, update the running summary
        if role == "assistant":
            # Split the existing summary (the system message) from the new conversation turns
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary = message["content"]
                else:
                    conversation += f"{message['role']}: {message['content']}\n"

            # Ask the LLM to extend the summary with the new conversation
            prompt = f"""Update the summary with the new conversation.

Summary: {summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            # A reply with no text (content is None) leaves the old summary in place
            self.messages = [{"role": "system", "content": response.content or summary}]
