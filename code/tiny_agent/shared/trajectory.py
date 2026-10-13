"""The sequence of steps an agent takes to reach its final answer.

To reach a final answer, an agent might need various steps and use different
actions (tool calls). This sequence of :class:`~tiny_agent.shared.step.Step`
objects is the agent's *trajectory*.

The trajectory is where each ``Step`` is created and filled with information
related to the TinyAgent. It is tracked because it makes the sequence easy to
debug: you can see where, and why, an agent failed.
"""

from llm_client.response import Response
from tiny_agent.shared.step import Step


class Trajectory:
    """Records agent execution as a sequence of runs.

    Role: the agent's logbook. It is for the developer reading afterwards, not
    for the model (what the model sees is ``tiny_agent.shared.memory.Memory``).

    Attributes:
        runs: One dict per call to ``TinyAgent.run``, oldest first, in the form
            ``{"query": <task text>, "steps": [Step, ...]}``.
    """
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query.

        Must be called before :meth:`add`, which writes to the latest run.
        """
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation.

        Args:
            response: What the model returned for this step.
            observation: What the tool returned, when the model's reply was a
                tool call that has been run. ``None`` means no tool ran, so the
                reply is treated as the final answer.

        Raises:
            IndexError: No run has been started with :meth:`initialize`.
        """
        # Add THOUGHT: the model's reasoning, or "" when it sent none
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)
