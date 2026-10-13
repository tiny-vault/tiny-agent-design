# Architecture overview

## Components

| Component | Source | Responsibility |
|---|---|---|
| Entry point | `main.py` (`main`) | Creates the agent, runs two tasks on it (the authors' names, then "what are our names?") to show memory, prints the second result. Holds no agent logic. |
| Harness | `tiny_agent/agent/tiny_agent.py` (`TinyAgent`) | Control loop and the slots for LLM, memory, tools and planner. |
| Memory | `tiny_agent/agent/memory.py` (`Memory`) and `tiny_agent/memory/` (`trimming.TrimmingMemory`; `summarization.SummarizationMemory`; `rag.RAGMemory`; `memory_bank.MemoryBank`) | The conversation history sent to the model on every call. `TrimmingMemory` keeps only the last two turns; `SummarizationMemory` keeps a model-written running summary instead of the messages; `RAGMemory` puts the documents most similar to each question in front of it (found with the embedding client `tiny_agent/llm_client/embedding.py`). `MemoryBank` keeps a summary per turn that fades unless recalled, plus a user portrait. `main.py` uses them when `TINYAGENT_TRIM_MEMORY`, `TINYAGENT_SUMMARY_MEMORY`, `TINYAGENT_RAG_MEMORY` or `TINYAGENT_BANK_MEMORY` is on. |
| Trajectory | `tiny_agent/agent/trajectory.py` (`Trajectory`, `Step`) | A debugging record of each run, one step at a time. |
| LLM client | `tiny_agent/llm_client/llm.py` (`LLM`) | Sends a conversation to an OpenAI-compatible server and returns a `Response`. Checks the model is loaded first and turns every failure into a clear `RuntimeError`. |
| Reply type | `tiny_agent/llm_client/response.py` (`Response`) | The answer text, the reasoning, the first tool call and metadata. |
| Model check | `tiny_agent/llm_client/check.py` | Asks the server which models are loaded (`python -m tiny_agent.llm_client.check`). |
| Terminal chat | `cli.py` | The `tiny-agent` command: interactive chat or one question. Writes `logs/cli.log`. |
| Docker deployment | `deploy/` | Four independent parts, each with its own compose file: the chat app, the llm client with its own model server, the tiny agent, the docs. `execute.mjs` starts and stops them. |

`TinyAgent` has five attributes. `llm` and `memory` are passed in when it is created, `trajectory` is created inside it, and `tools` and `planner` are `None` until their phases. `_step` calls the LLM client with the whole memory, so the harness already talks to a model. The terminal chat does not use `TinyAgent`: it keeps its own message list.

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

Solid lines exist in the code now. Dashed lines are the intended loop: `_step` never calls `_execute_action` yet. `run` adds the task to memory; `_step` sends the memory to the model, stores the reply and records a step in the trajectory.

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
Tool dispatch, planning, the multi-step loop, trimming the memory and
configuration. None of these exist in `TinyAgent` yet.
```

Besides the Python package there is a separate Docker chat app, described in {doc}`../guides/chat-app`. The Python code reads these settings from the environment: `TINYAGENT_MODEL` and `TINYAGENT_LLM_URL` (defaults for the chat command and the demo), `TINYAGENT_TRIM_MEMORY` (the demo's memory keeps only the last two turns), `TINYAGENT_SUMMARY_MEMORY` (the demo's memory keeps a running summary), `TINYAGENT_API_KEY` or `TINYAGENT_API_KEY_FILE` (for a server that needs a key), and `TINYAGENT_LOG_DIR` and `TINYAGENT_LOG_LEVEL` (the log files). The `LLM` client itself reads nothing from the environment.
