# The agent harness

A language model only turns text into text. It cannot remember the last message, run code or look at a file. An *agent* is that model plus the code around it, the **harness**: something that keeps the conversation, decides what to send, runs the tools the model asks for, and records what happened. `TinyAgent` ({py:class}`tiny_agent.agent.TinyAgent`) is that harness, built up one phase at a time.

```{mermaid}
flowchart LR
    T["task"] --> M["memory"]
    M --> L["LLM decides"]
    L -->|answer| A["result"]
    L -->|tool call| X["tool"]
    X -->|observation| M
```

The model decides; the harness does. Everything in the diagram except the box labelled *LLM decides* is harness code.

## The five slots

`TinyAgent(llm, memory, tools=None)` is wiring. Each slot is a part you can swap without touching the others.

| Slot | Role | What fills it | Page |
|---|---|---|---|
| `llm` | The brain: chooses the next step. | `LLM` | {doc}`model-client` |
| `memory` | The context: what the model sees on each call. | a `BaseMemory` | {doc}`memory` |
| `tools` | The hands: functions the agent may call. | a `BaseTools`, or `None` for a plain chat | {doc}`tools` |
| `trajectory` | The logbook: what happened, for you. | `Trajectory`, created inside the agent | {doc}`trajectory` |
| `planner` | The strategy: break a goal into steps. | `None`, a `ReAct` (`components/planning/react.py`) or a `NativeReAct` (`components/planning/native_react.py`) | |

## What `run` does today

1. `run(task)` stores the task in `memory` and starts a run in the `trajectory`.
2. `_step()` sends the whole memory to the model and stores the reply in memory.
3. With no tools, the reply is the answer. With tools, `tools.parse` looks for a tool call in the reply. None means the reply is the answer.
4. A tool call goes to `_execute_action`, which runs the tool, stores `OBSERVATION: <result>` in memory (a `user` message, or a `tool` message after a structured call) and records the step.

That is one model call per task. Because the history is resent every time, a second `run` on the same agent can use what the first one said; with a fresh memory it cannot.

```{note}
**The loop (opt-in):** with `planner=ReAct(max_steps=10)` the system prompt teaches `THOUGHT:` / `ACTION:` and `run` repeats step, act, observe until the model calls `final_answer` (or answers in plain text), giving up with `"Max steps reached without completion."`. `ReAct.parse` puts the thought into `reasoning` (the trajectory's `thought`) and the action into `content` for the tools to read; it needs prompting tools. Without a planner `run` is still one model call.
```

## Memory versus trajectory

They hold similar facts for different readers. `memory` is what the *model* sees: the messages sent on every call, and what a memory variant may trim or summarise. `trajectory` is what *you* see afterwards: a complete record of each run, never sent to the model. Debugging an agent that used a tool wrongly means reading the trajectory, not the memory.

## Why it is built this way

* **Plain Python, one file.** The harness is `tiny_agent/agent.py`, about 150 lines. Nothing hides in a framework.
* **Nothing is forced.** The agent takes whichever memory and tools it is given, so every variant in this book runs through the same harness.
* **The model client is separate.** The agent asks `llm.generate(messages)` and gets a `Response`; it never sees server JSON. The terminal chat uses the same client without the agent (see {doc}`overview`).
