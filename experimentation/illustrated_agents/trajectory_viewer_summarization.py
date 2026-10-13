"""The summarizing-memory agent from ``summarization_memory_first_turn``, followed by a look at its trajectory.

**What it shows.** The ``Trajectory`` is the agent's logbook for the developer; the memory is what the model sees.
With ``SummarizationMemory`` the two differ sharply: the memory shrinks to a one-message summary, but the
trajectory still records the run. In the book, a notebook cell ``TrajectoryViewer(agent.trajectory)`` draws it as
collapsible HTML. This script prints the plain-text equivalent, because the viewer only renders inside Jupyter.

**How it works here.** ``llm`` is the model, ``memory`` a ``SummarizationMemory`` built with it and ``agent`` a
``TinyAgent`` built with both. ``query`` is sent with one ``agent.run``, then ``viewer`` wraps ``agent.trajectory``
and ``show_trajectory`` prints the run with its steps. A step with an ``answer`` is a final answer; one with an
``action`` is a tool call; otherwise it is just thinking.

**What to expect.** One run with one step, a final answer. The trajectory shows only the agent's own model call:
the call that writes the summary is made by the memory, so it appears in neither the trajectory nor the messages.
Replies vary between runs. ``llm`` is ``qwen3.5-4b`` via the gateway with the server's default temperature.
Compare ``trajectory_viewer``, which does the same for the trimming-memory agent.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch4 import SummarizationMemory, TinyAgent
from illustrated_agents.utils import TrajectoryViewer

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's calls: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

query = "Hi, my name is Sarah. Why are flamingos pink?"


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
    # Create the agent and run a query (needs the Docker stack with qwen3.5-4b loaded)
    memory = SummarizationMemory(llm=llm)
    agent = TinyAgent(llm=llm, memory=memory)
    agent.run(query)

    # The notebook cell: TrajectoryViewer(agent.trajectory)
    viewer = TrajectoryViewer(agent.trajectory)
    print(f"memory holds {len(memory.get_messages())} message(s); trajectory holds {len(viewer.trajectory.runs)} run(s)\n")
    show_trajectory(viewer.trajectory)
