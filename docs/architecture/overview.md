# Architecture overview

## Components

| Component | Source | Responsibility |
|---|---|---|
| Entry point | `code/main.py` (`main`) | Creates the agent, runs one task, prints the result. Holds no agent logic. |
| Harness | `code/tiny_agent/shared/tiny_agent.py` (`TinyAgent`) | Control loop and the slots for LLM, memory, tools and planner. |
| LLM client | `code/llm_client/llm.py` (`LLM`) | Sends a conversation to an OpenAI-compatible server and returns a `Response`. Checks the model is loaded first and turns every failure into a clear `RuntimeError`. |
| Reply type | `code/llm_client/response.py` (`Response`) | The answer text, the reasoning, the first tool call and metadata. |
| Model check | `code/llm_client/check.py` | Asks the server which models are loaded (`python -m llm_client.check`). |
| Terminal chat | `code/cli.py` | The `tiny-agent` command: interactive chat or one question. Writes `logs/cli.log`. |

`TinyAgent` has four attributes, all `None` today and filled in by later steps: `llm`, `memory`, `tools` and `planner`. The LLM client exists and works, but it is not connected to `TinyAgent` yet: the chat command and the harness are separate.

## Data flow of the harness

```{mermaid}
flowchart LR
    M["main.main()"] -->|task| R["TinyAgent.run"]
    R --> S["TinyAgent._step"]
    S -->|answer| R
    S -.->|action, planned| E["TinyAgent._execute_action"]
    E -.->|observation, planned| S
    R -->|result| M
```

Solid lines exist in the code now. Dashed lines are the intended loop: `_step` never calls `_execute_action` yet, and `_step` currently returns `"Received: <task>"` without consulting any model.

## Data flow of the terminal chat

```{mermaid}
flowchart LR
    U["tiny-agent"] --> P["pick_model: which model is loaded?"]
    P --> L["LLM.generate"]
    L -->|"HTTP /v1"| G["gateway, backend, llama.cpp"]
    G -->|"Response"| U
    U -.->|"cli.log"| O["Vector, then OpenObserve"]
```

The chat sends the whole conversation on every turn, so the model remembers it. If a turn fails or is cancelled, the unanswered question is dropped from the history, because many chat templates reject two user messages in a row.

## Harness roles

* `run` is the public entry point and will become the outer loop with a stop condition.
* `_step` is the "think" phase: build the prompt, ask the LLM, return an answer or an action.
* `_execute_action` is the "act" phase: dispatch to a tool and return an observation.

## Not yet implemented

```{todo}
Connecting the LLM client to the harness, memory, tool dispatch, planning, the
multi-step loop and configuration. None of these exist in `TinyAgent` yet.
```

Besides the Python package there is a separate Docker chat app, described in {doc}`../guides/chat-app`. The only configuration the Python code reads from the environment is `TINYAGENT_MODEL` and `TINYAGENT_LLM_URL` (defaults for the chat command) and `TINYAGENT_LOG_DIR` and `TINYAGENT_LOG_LEVEL` (the log files). The `LLM` client itself reads nothing from the environment.
