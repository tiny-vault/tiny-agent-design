# Trajectory and steps

To reach a final answer an agent may need several steps and several tool calls, and sometimes it uses a tool wrongly. To find out why, the harness records everything it did. That record is the **trajectory**, and each entry in it is a **step**.

It is the agent's logbook: written for the developer reading afterwards, never sent to the model. What the model sees is the {doc}`memory`.

## The shape

```text
trajectory.runs = [
    {"query": "What is 6 times 7?",
     "steps": [Step(...), Step(...)]},
    {"query": "And times 2?",
     "steps": [Step(...)]},
]
```

`TinyAgent.run` starts one entry in `runs` for every task. Each model call adds a `Step` ({py:class}`tiny_agent.components.step.Step`) to the latest run.

| `Step` field | Holds |
|---|---|
| `thought` | What the model reasoned before acting (`""` if it sent none). |
| `action` | The tool call the model made, or `None`. |
| `observation` | What the tool returned, or `None` if no tool ran. |
| `answer` | The final answer. When it is set, the agent has reached the end of its turn. |
| `metadata` | Bookkeeping: the model and the token counts. |

*Action* is the book's word for a tool call, and *observation* for its result.

## Two kinds of step

`Trajectory.add(response, observation=None)` fills a step from a `Response` ({doc}`model-client`):

* **A tool ran.** `observation` is given, so the step holds the `action` and the `observation`.
* **No tool ran.** `observation` is `None`, so the reply is treated as the final answer and the step holds the `answer`.

So a run with one tool call is two steps: the call and its observation, then the answer, once the loop exists. Today, with one model call per task, a run that uses a tool ends after its first step.

## Reading it

```python
agent.run("What is 6 times 7?")
for run in agent.trajectory.runs:
    print(run["query"])
    for step in run["steps"]:
        print(step.thought, step.action, step.observation, step.answer)
```

Scripts in the book's `illustrated_agents` folder (`trajectory_viewer.py`) print a trajectory this way. {py:class}`~tiny_agent.components.trajectory.Trajectory` is the class that holds it.

## Memory or trajectory?

| | Memory | Trajectory |
|---|---|---|
| Read by | the model | you |
| Sent to the model | yes, every call | never |
| May be shortened | yes (trimming, summaries) | no, it is the full record |
