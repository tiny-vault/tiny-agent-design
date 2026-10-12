"""Run the book's chapter 2 ``TinyAgent`` on one question and print its trajectory.

What it does:
    Builds the chapter 2 ``TinyAgent`` around :data:`llm`, asks it "What is 2 + 2?", and prints three things:

    1. The agent's final answer (the return value of ``agent.run``).
    2. ``agent.trajectory.runs``, the raw record: a list with one dict per ``run`` call, each holding the user ``query`` and its list of ``steps``.
    3. A readable version of the same trajectory, which is what ``TrajectoryViewer`` would render as HTML in Jupyter: one block per run, one line per step, labelled ``Final answer``, the name of the action taken, or ``Thinking``, followed by every non-empty field of the step (``THOUGHT``, ``ACTION``, ``OBSERVATION``, ``ANSWER``, ``METADATA``) in upper case.

    A trajectory is the agent's step-by-step history: what it thought, which action it took, what it observed and when it answered. For a question this easy the chapter 2 agent usually answers in a single step, so expect one run with one ``Final answer`` step such as "The answer is **4**."

Run:
    ``.venv/Scripts/python trigger_ias.py run_agent_trajectory`` from the repo root.

Requires:
    The Docker stack up with ``qwen3.5-4b`` loaded (``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``). Importing the module makes no network call; only the entry point runs the agent.
"""

from illustrated_agents.chapters.ch2 import LLM, TinyAgent
from illustrated_agents.utils import TrajectoryViewer

# The chat-app gateway forwards /v1 to llama.cpp and adds the API key itself.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1")
"""The model client, ``qwen3.5-4b`` through the chat-app gateway at ``http://localhost:3200/v1``.

It passes no API key (the client's placeholder is sent); the gateway accepts the call without one. Default temperature.

:meta hide-value:
"""

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
