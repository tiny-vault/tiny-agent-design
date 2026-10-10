# Architecture overview

## Components

| Component | Source | Responsibility |
|---|---|---|
| Entry point | `main.py` (`main`) | Creates the agent, runs one task, prints the result. Holds no agent logic. |
| Harness | `tiny_agent.py` (`TinyAgent`) | Control loop and the slots for LLM, memory, tools and planner. |

`TinyAgent` has four attributes, all `None` today and filled in by later
steps: `llm`, `memory`, `tools` and `planner`.

## Data flow

```{mermaid}
flowchart LR
    M["main.main()"] -->|task| R["TinyAgent.run"]
    R --> S["TinyAgent._step"]
    S -->|answer| R
    S -.->|action, planned| E["TinyAgent._execute_action"]
    E -.->|observation, planned| S
    R -->|result| M
```

Solid lines exist in the code now. Dashed lines are the intended loop:
`_step` never calls `_execute_action` yet, and `_step` currently returns
`"Received: <task>"` without consulting any model.

## Harness roles

* `run` is the public entry point and will become the outer loop with a stop
  condition.
* `_step` is the "think" phase: build the prompt, ask the LLM, return an
  answer or an action.
* `_execute_action` is the "act" phase: dispatch to a tool and return an
  observation.

## Not yet implemented

```{todo}
LLM integration, memory, tool dispatch, planning, the multi-step loop, error
handling and configuration. None of these exist in the code yet.
```

The project has no external integrations, configuration files, or
authentication.
