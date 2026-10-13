"""The trimming-memory agent from ``agent_with_trimming_memory``, followed by a look at its trajectory.

**What it shows.** Memory is what the *model* sees; the ``Trajectory`` is the agent's logbook for the *developer*:
one entry per ``run``, each holding the ``query`` and the ``Step`` objects the agent took. In the book, a notebook
cell ``TrajectoryViewer(agent.trajectory)`` draws it as collapsible HTML. This script runs the same agent and prints
the plain-text equivalent, because the viewer only renders inside Jupyter.

**How it works here.** ``llm`` is the model, ``memory`` a ``TrimmingMemory`` and ``agent`` a ``TinyAgent`` built with
both. The four ``tasks`` run one after another, then ``viewer`` wraps ``agent.trajectory`` and ``show_trajectory``
prints every run with its steps. A step with an ``answer`` is a final answer; one with an ``action`` is a tool call;
otherwise it is just thinking. There are no tools yet, so every step here is a final answer.

**What to expect.** Four runs with one step each. The trajectory keeps all four queries and answers even though
``memory`` only holds the last two turns: trimming forgets for the model, not for the logbook. Replies vary between
runs. ``llm`` is ``qwen3.5-4b`` via the gateway with the server's default temperature. Compare
``agent_with_trimming_memory``, which prints the memory instead.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch4 import TinyAgent, TrimmingMemory
from illustrated_agents.utils import TrajectoryViewer

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's calls: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

tasks = [
    "Hi! I'm reading 'An Illustrated Guide to AI Agents'.",
    "There are many flamingos in this book, why?",
    "What is 1+1?",
    "What is 2+2?",
]


def show_trajectory(trajectory):
    """Print a trajectory as indented text: each run's query, then its steps and their non-empty fields.

    This is the plain-text version of what ``TrajectoryViewer`` draws as HTML.

    Args:
        trajectory: A ``Trajectory`` whose ``runs`` are ``{"query": ..., "steps": [Step, ...]}``.
    """
    for run_index, run in enumerate(trajectory.runs, start=1):
        print(f"Run {run_index} - user query: {run['query']}")
        for step_index, step in enumerate(run["steps"], start=1):
            # Same labelling as the viewer: answer, else the tool called, else just thinking
            if step.answer:
                kind = "Final answer"
            elif step.action:
                kind = step.action.get("name", "action")
            else:
                kind = "Thinking"
            print(f"  Step {step_index} - {kind}")
            for label, value in vars(step).items():
                if value:
                    # Only the start of long values, enough to recognise the step
                    print(f"    {label.upper()}: {str(value)[:100]!r}")
        print()


if __name__ == "__main__":
    # One agent, four runs (needs the Docker stack with qwen3.5-4b loaded)
    memory = TrimmingMemory()
    agent = TinyAgent(llm=llm, memory=memory)
    for task in tasks:
        agent.run(task)

    # The notebook cell: TrajectoryViewer(agent.trajectory)
    viewer = TrajectoryViewer(agent.trajectory)
    print(f"memory holds {len(memory.get_messages())} message(s); trajectory holds {len(viewer.trajectory.runs)} run(s)\n")
    show_trajectory(viewer.trajectory)
