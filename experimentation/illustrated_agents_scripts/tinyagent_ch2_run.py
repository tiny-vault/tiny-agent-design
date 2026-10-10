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
