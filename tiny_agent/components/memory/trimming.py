"""Trimming memory: keep only the most recent part of the conversation.

WHAT IS IT? ``Memory`` keeps every message and resends all of them on every model call, so a long chat
eventually passes the model's context window (and gets slower and more expensive as it grows).
``TrimmingMemory`` is the simplest fix: after each ``add`` it throws away everything except the system
message(s) and the last four other messages, which is the last two turns (a user message and the
assistant's reply, twice). That is a sliding window over the conversation.

HOW IT WORKS: ``add`` stores the new message with the parent class, then rebuilds ``messages`` as the
system messages (always kept, always first) followed by the last four non-system messages. Nothing is
summarised or saved: what falls out of the window is gone, so the agent forgets it.

WHAT TO NOTICE: the window is counted in messages, not in tokens, so a few very long messages can still
be large. A ``tool`` message counts as one of the four, like any other. And because the trim happens when
the new user message is added, the model call that follows can see a window that starts with an assistant
message (``[assistant, user, assistant, user]``); the first turn is cut in the middle. Most chat templates
accept that, but a stricter one might not.

Switch it on in the demo with ``TINYAGENT_TRIM_MEMORY=1`` (see ``main.memory_from_env``); the harness
itself takes whichever memory it is given.
"""

from typing import Any, override

from tiny_agent.components.memory.memory import BaseMemory


class TrimmingMemory(BaseMemory):
    """Memory that keeps only the last two user/assistant turns.

    Role: a drop-in replacement for ``Memory`` in the harness's "context" slot, for chats that should not
    grow without limit. It has the same methods; only ``add`` differs.

    Attributes:
        messages: The system message(s), then at most the four most recent other messages, oldest first,
            plus the pinned message when there is one.
    """

    _pinned: dict[str, Any] | None = None

    @override
    def add(self, role: str, content: str, tool_call: dict[str, Any] | None = None, **kwargs) -> None:
        """Add a message, then drop everything but the system message(s) and the last four others.

        Args:
            role: Who said it: ``system``, ``user``, ``assistant`` or ``tool``.
            content: The message text.
            tool_call: The tool call the model made, if this message records one (stored as in ``Memory``).
            **kwargs: ``pin=True`` keeps this message outside the window until a later message is
                pinned (the agent pins the task of a ReAct loop, so a long loop does not forget it).
                Anything else is ignored, as in ``Memory``.
        """
        # Add the new message first using the parent class's helper (BaseMemory)
        self._append(role, content, tool_call)
        if kwargs.get("pin"):
            self._pinned = self.messages[-1]

        # Then keep the system message(s), the pinned one and the most recent two turns (4 messages)
        turns = [m for m in self.messages if m["role"] != "system" and m is not self._pinned]
        recent = turns[-4:]
        system = [m for m in self.messages if m["role"] == "system"]
        kept = [m for m in self.messages if m is self._pinned or any(m is r for r in recent)]
        self.messages = system + kept
