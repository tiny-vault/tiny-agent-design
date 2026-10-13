"""Run the chapter 2 ``TinyAgent`` on "What is 2 + 2?" and print its trajectory.

Prints the answer, the raw ``agent.trajectory.runs``, then a plain-text version of what ``TrajectoryViewer`` renders in Jupyter. Expect one run with one ``Final answer`` step. ``llm`` is ``qwen3.5-4b`` via the gateway, which accepts the call without an API key.
"""

from illustrated_agents.chapters.ch2 import LLM, TinyAgent
from illustrated_agents.utils import TrajectoryViewer

from common import agent_query

# The chat-app gateway forwards /v1 to llama.cpp and adds the API key itself.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1")

if __name__ == "__main__":
    agent = TinyAgent(llm=llm)
    response = agent.run(agent_query)
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
