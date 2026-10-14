# The model client

The agent never talks to a model server directly. It talks to a client that turns a list of messages into a `Response`. The client is small and has no dependency on the rest of the harness, so it also works on its own, as the terminal chat and the book's scripts show.

```{mermaid}
flowchart LR
    A["messages"] --> C["check: is the model loaded?"]
    C --> B["build the request body"]
    B --> P["POST /v1/chat/completions"]
    P --> R["Response"]
```

## Three pieces

| Piece | File | What it does |
|---|---|---|
| `LLM` | `components/client/llm.py` | One chat model on one server. `generate(messages, tools=None)` returns a `Response`. |
| `EmbeddingModel` | `components/client/embedding.py` | One embedding model. `embed(text)` returns a vector (a list of floats), used by `RAGMemory` and `MemoryBank`. |
| `Response` | `components/client/response.py` | The flat shape of a reply: `content`, `reasoning`, `tool_call`, `metadata`. |

Both clients send their request with `post_json` (`http.py`), which turns every failure into one `RuntimeError` that says what went wrong: the server cannot be reached, it answered with an error status, or the reply is not JSON.

## Any OpenAI-compatible server

The client speaks the `/v1/chat/completions` dialect that most local and hosted servers share. The default address is the Docker gateway, `http://127.0.0.1:3200/v1`; point it at Ollama with `base_url="http://localhost:11434/v1"`. Model names are whatever the server knows: `gemma4-e4b` on the gateway, `gemma4:e4b-it-qat` on Ollama.

## What a `Response` is

One reply is either a **final answer** (`content` is set) or a **request to run a tool** (`tool_call` is set and `content` is often `None`). The harness branches on which one it got.

| Field | Holds |
|---|---|
| `content` | The answer text, or `None` when the model only asked for a tool. |
| `reasoning` | The model's thinking, when it sends it separately and `think` is on. llama.cpp calls it `reasoning_content`, Ollama `reasoning`; the client reads either. |
| `tool_call` | The first tool call, as the server sent it. `arguments` is a JSON *string*. Any further calls are dropped. |
| `metadata` | `model` (which model answered), `prompt_tokens`, `completion_tokens`. |

Converting once, here, keeps the differences between servers out of the agent.

## The loaded-model check

A 6 GB GPU holds one model at a time. The Docker gateway runs llama.cpp in *router mode*: it offers several models, keeps one loaded and refuses to chat with the others. Before each request `generate` asks `GET /models` which are loaded, and decides:

| The server says | `generate` does |
|---|---|
| the model is `loaded` | uses it |
| it is `unloaded` or `loading` | raises a `RuntimeError` with the fix: `node execute.mjs load <model>` |
| the model is not offered at all | uses a model that *is* loaded, and warns |
| it reports no status (Ollama), or cannot be reached | never blocks the request |

You can ask yourself: `python -m tiny_agent.components.client.check` lists every model and its status, and `check <model>` exits 0 if it is loaded, 1 if not, 2 if the server is down.

## Limits

The client is blocking (no streaming), makes no retries and keeps no history: the caller owns the conversation, which is exactly why the agent has a memory. The `/models` pre-check costs one extra request per call, and a call gives up after 600 seconds, long enough for a slow local model to reason.
