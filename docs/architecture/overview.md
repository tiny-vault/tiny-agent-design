# Architecture overview

## Components

| Component | Source | Responsibility |
|---|---|---|
| Entry point | `main.py` (`main`) | Creates the agent, runs two tasks on it (the authors' names, then "what are our names?") to show memory, prints the second result. Holds no agent logic. |
| Harness | `tiny_agent/agent.py` (`TinyAgent`) | Control loop and the slots for LLM, memory, tools and planner. |
| Memory | `tiny_agent/components/memory/` (`Memory`, `TrimmingMemory`, `SummarizationMemory`, `RAGMemory`, `MemoryBank`) | The conversation history sent to the model on every call. Five variants share one `BaseMemory`; see {doc}`memory`. `main.py` picks one from the `TINYAGENT_*_MEMORY` settings. |
| Tools | `tiny_agent/components/tools/` (`PromptingTools`, `LearnedTools`, `multiply`) | The functions the agent may call, and how a call is read and run; see {doc}`tools`. |
| Trajectory | `tiny_agent/components/trajectory.py` (`Trajectory`), `tiny_agent/components/step.py` (`Step`) | A debugging record of each run, one step at a time. |
| LLM client | `tiny_agent/components/client/llm.py` (`LLM`) | Sends a conversation to an OpenAI-compatible server and returns a `Response`. Checks the model is loaded first and turns every failure into a clear `RuntimeError`. |
| Reply type | `tiny_agent/components/client/response.py` (`Response`) | The answer text, the reasoning, the first tool call and metadata. |
| Model check | `tiny_agent/components/client/check.py` | Asks the server which models are loaded (`python -m tiny_agent.components.client.check`). |
| Terminal chat | `cli.py` | The `tiny-agent` command: interactive chat or one question. Writes `logs/cli.log`. |
| Settings and logs | `tiny_agent/config/`, `tiny_agent/logging/log.py` | `Settings` reads every `TINYAGENT_*` value (environment, then the `.env` file, then a default). `log.py` writes `cli.log` and `main.log`. |
| Docker deployment | `deploy/` | Four independent parts, each with its own compose file: the chat app, the llm client with its own model server, the tiny agent, the docs. `execute.mjs` starts and stops them. |

`TinyAgent` has five attributes. `llm`, `memory` and (optionally) `tools` are passed in when it is created, `trajectory` is created inside it, and `planner` is `None` until its phase. `_step` calls the LLM client with the whole memory, so the harness already talks to a model, and when the reply holds a tool call it hands it to `_execute_action`. The terminal chat does not use `TinyAgent`: it keeps its own message list.

## Data flow of the harness

```{mermaid}
flowchart LR
    M["main.main()"] -->|task| R["TinyAgent.run"]
    R --> S["TinyAgent._step"]
    S -->|answer| R
    S -->|tool call| E["TinyAgent._execute_action"]
    E -->|observation| S
    R -->|result| M
```

`run` adds the task to memory; `_step` sends the memory to the model, stores the reply and records a step in the trajectory. If the reply holds a tool call, `_execute_action` runs it and stores the observation in memory. `_step` returns that observation; it does not call the model again, because the multi-step loop is not built yet.

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
* `_step` is the "think" phase: send the memory to the LLM, store its reply, return an answer or an action.
* `_execute_action` is the "act" phase: dispatch to a tool and return an observation.

## Not yet implemented

```{todo}
Planning and the multi-step loop (call the model again with the observation until it answers). Neither exists in `TinyAgent` yet.
```

Besides the Python package there is a separate Docker chat app, described in {doc}`../guides/chat-app`. The Python code reads its settings from the environment or a `.env` file (see `tiny_agent/.env.example`):

| Setting | What it does |
|---|---|
| `TINYAGENT_MODEL`, `TINYAGENT_LLM_URL` | The chat model and the server address (defaults for the chat command and the demo). |
| `TINYAGENT_API_KEY` or `TINYAGENT_API_KEY_FILE` | The key, for a server that needs one. |
| `TINYAGENT_THINK`, `TINYAGENT_SYSTEM` | Let the model reason; set a system message. |
| `TINYAGENT_TRIM_MEMORY`, `TINYAGENT_SUMMARY_MEMORY`, `TINYAGENT_RAG_MEMORY`, `TINYAGENT_BANK_MEMORY` | Which memory the demo uses (the strongest wins). |
| `TINYAGENT_EMBEDDING_MODEL`, `TINYAGENT_EMBEDDING_URL` | The embedding model for the `rag` and `bank` memories. |
| `TINYAGENT_TOOLS`, `TINYAGENT_NATIVE_TOOLS` | Give the agent the built-in tools, prompted or learned. |
| `TINYAGENT_LOG_DIR`, `TINYAGENT_LOG_LEVEL` | Where the log files go and how much they say. |

The `LLM` client itself reads nothing from the environment.
