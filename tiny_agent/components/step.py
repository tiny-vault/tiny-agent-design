"""A record of one thing the agent did, kept so a run can be debugged afterwards.

An agent may run for several turns and sometimes uses a tool wrongly. To find out
why, the harness tracks everything the model has done so far, one :class:`Step`
at a time, including what each tool returned.

``Step`` is a pydantic data container, not the agent's control flow. All fields
have defaults, so an empty ``Step()`` is valid. ``Trajectory.add`` fills the
thought and metadata, then either action plus observation or answer.
The distinction separates a model request from the consequence of running it.

Inspecting a step shows recorded text and tool results; it does not expose
internal model computation or prove that an answer is correct. ``thought``
is separately returned reasoning text when available. This file performs no
network requests, printing or file writes. Compare ``response.py`` for the
model reply before the harness adds its outcome.
"""

from typing import Any

from pydantic import BaseModel


class Step(BaseModel):
    """A single step in an agent's trajectory.

    A step is very much like :class:`tiny_agent.components.client.response.Response`, with one major
    difference: it adds the ``answer`` and ``observation`` fields. These are
    consequences of processing, results we get from the TinyAgent rather than
    something the model sent directly. An *action* is the term for a tool call.

    Attributes:
        thought: What the model reasoned before acting.
        action: The tool call the model made (a tool name and its arguments), or
            ``None`` if this step called no tool.
        observation: The output of the tool that ran for this step's action, or
            ``None`` if no tool ran.
        answer: The model's final answer. When it is set, the agent has reached
            the end of its turn.
        metadata: Bookkeeping about the call, such as the model and token counts.
    """

    thought: str = ""
    action: dict[str, Any] | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict[str, Any] | None = None
