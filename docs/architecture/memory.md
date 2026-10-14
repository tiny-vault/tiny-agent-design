# Memory

A model server is stateless: it forgets everything between requests. An agent that should remember what was said therefore keeps the history itself and sends all of it with every request. That stored list of messages is the agent's memory, and it is the slot where most of the interesting design choices live.

Every memory is a `BaseMemory` ({py:class}`tiny_agent.components.memory.memory.BaseMemory`): a list of chat messages (`{"role": ..., "content": ...}`), `get_messages()` to read them back before each model call, and `add()` to store a new one. What differs between memories is only what `add` does.

```{mermaid}
flowchart LR
    T["task"] --> A["memory.add('user', task)"]
    A --> G["memory.get_messages()"]
    G --> L["LLM.generate"]
    L --> R["memory.add('assistant', reply)"]
```

## The five memories

| Memory | Keeps | Cost per turn | Switch (demo) |
|---|---|---|---|
| `Memory` | Every message, forever. | None. | the default |
| `TrimmingMemory` | The system message and the last two turns (four messages). | None. | `TINYAGENT_TRIM_MEMORY=1` |
| `SummarizationMemory` | One `system` message holding a running summary the model rewrites after each reply. | One extra model call. | `TINYAGENT_SUMMARY_MEMORY=1` |
| `RAGMemory` | The whole conversation, and puts the three documents most similar to each question in front of it. | One embedding request per user message. | `TINYAGENT_RAG_MEMORY=1` |
| `MemoryBank` | Per-turn summaries that fade unless recalled, plus a portrait of the user. | Two chat calls and one embedding request. | `TINYAGENT_BANK_MEMORY=1` |

When several switches are on, the strongest wins: bank, then rag, then summary, then trim. `rag` and `bank` need a chat model and an embedding model at the same time (`LLAMA_MODELS_MAX=2`).

## What each one teaches

```{tip}
Read them in this order. Each file in `tiny_agent/components/memory/` opens with a *what is it / how it works / what to notice* docstring.
```

* **Plain memory** is the baseline, and its limit is the context window: a long chat eventually no longer fits.
* **Trimming** is the simplest fix, a sliding window. What falls out is gone, and the window is counted in messages, not tokens.
* **Summarization** trades a model call for a longer reach: the facts the summary captured survive, and whatever it leaves out is lost.
* **RAG** adds knowledge that was never said in the conversation, by embedding the documents once and ranking them against each question with cosine similarity (`similarity.py`).
* **MemoryBank** builds its own long-term store from the conversation and lets old memories fade along an Ebbinghaus curve, `retention = exp(-t / S)`. Each recall adds to `S`, so what is used often lasts longer.

## Writing your own

Subclass `BaseMemory`, write `add`, and finish it with `self._append(role, content, tool_call)`. Pass the instance to `TinyAgent` as `memory`; the agent does not care which one it gets.
