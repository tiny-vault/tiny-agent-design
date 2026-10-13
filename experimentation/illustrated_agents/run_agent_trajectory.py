"""Run the phase 2 ``TinyAgent`` on "What is 2 + 2?" and print its trajectory.

**What it is.** A trajectory is the agent's record of what happened during a run: the user's query, then each step it took (a thought, an action or a final answer). It lets you inspect how the agent got its answer, not just the answer.

**How it works here.** The phase 2 agent makes one LLM call, so a run has one step. The script prints the answer, the raw ``agent.trajectory.runs`` (a list of runs, each with its ``query`` and ``steps``), then a plain-text version of what ``TrajectoryViewer`` renders as HTML in Jupyter: each step's kind and its non-empty fields.

**What to expect.** One run with one ``Final answer`` step; the reply text varies between runs. ``llm`` is ``qwen3.5-4b`` via the gateway, which accepts the call without an API key here because the backend adds the key to ``/v1`` requests.

Compare ``agent_greeting``, which runs the same agent and prints only the reply.
"""

from illustrated_agents.chapters.ch2 import LLM, TinyAgent
from illustrated_agents.utils import TrajectoryViewer

# The chat-app gateway forwards /v1 to llama.cpp and adds the API key itself.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1")

if __name__ == "__main__":
    agent = TinyAgent(llm=llm)
    response = agent.run("What is 2 + 2?")
    print(response)
    print(agent.trajectory.runs)

    # TrajectoryViewer(agent.trajectory) only renders HTML inside Jupyter, so
    # print the same runs and steps as plain text when run as a script.
    viewer = TrajectoryViewer(agent.trajectory)
    for run_index, run in enumerate(viewer.trajectory.runs, start=1):
        print(f"\nRun {run_index} - User query\n  {run['query']}")
        for step_index, step in enumerate(run["steps"], start=1):
            kind = "Final answer" if step.answer else (
                step.action.get("name", "action") if step.action else "Thinking")
            print(f"  Step {step_index} - {kind}")
            for label, value in vars(step).items():
                if value:
                    print(f"    {label.upper()}: {value}")
