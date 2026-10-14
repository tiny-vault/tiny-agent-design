"""Tests for ``Step`` and ``Trajectory``."""

from tiny_agent.components.client.response import Response
from tiny_agent.components.step import Step
from tiny_agent.components.trajectory import Trajectory


def test_step_defaults():
    step = Step()
    assert (step.thought, step.action, step.observation, step.answer, step.metadata) == ("", None, None, None, None)


def test_trajectory_records_steps_as_step():
    trajectory = Trajectory()
    trajectory.initialize("q")
    trajectory.add(Response(content="a"))
    (step,) = trajectory.runs[0]["steps"]
    assert type(step) is Step and step.answer == "a"
