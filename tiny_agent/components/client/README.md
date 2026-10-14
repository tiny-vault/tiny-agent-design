# client

A small, self-contained client for chat models behind an OpenAI-compatible API (llama.cpp, Ollama, and similar). It uses httpx for the requests.

| File | What it is |
|---|---|
| `http.py` | `post_json`, the one HTTP POST that `LLM` and `EmbeddingModel` both use. |
| `llm.py` | The `LLM` class. `generate(messages, tools=None)` sends a conversation and returns a `Response`. |
| `embedding.py` | The `EmbeddingModel` class. `embed(text)` sends a text to `/v1/embeddings` and returns the vector (a list of floats). |
| `response.py` | The `Response` dataclass: `content`, `reasoning`, `tool_call`, `metadata`. |
| `check.py` | Asks the server which models are loaded. Also a command: `python -m tiny_agent.components.client.check`. |
| `env.py` | Loads settings from a `.env` file (`load_env()`) and reads the API key (`api_key_from_env()`); used by the `tiny-agent` command, `main.py` and `check.py`, not by `LLM` itself. |
| `tests/` | pytest suite (123 tests) against a fake model server (no real model needed). |

## Use it

```python
from tiny_agent.components.client.llm import LLM

llm = LLM(model="gemma4-e4b")   # base_url defaults to http://127.0.0.1:3200/v1
reply = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
print(reply.content)
print(reply.metadata)           # model, prompt_tokens, completion_tokens
```

Point it at another server with `base_url`, for example Ollama: `LLM(model="gemma3:4b", base_url="http://localhost:11434/v1")`.

## What `generate` does for you

* On a llama.cpp router it asks which models are loaded first. A model that is offered but `unloaded` or `loading` raises a `RuntimeError` that says how to load it. A model the server does not offer at all is replaced, for that call only, by the loaded one, with a `UserWarning`. A server that reports no status (Ollama) is never blocked.
* Every failure (server down, HTTP error, a reply that is not JSON or not a chat completion, an empty `messages` list) is one clear error: `RuntimeError`, or `ValueError` for an empty `messages`.
* It gives up after 600 seconds.
* Reasoning is read from either `reasoning` (Ollama) or `reasoning_content` (llama.cpp). Only the first tool call is kept.

## In Docker, with its own model server

`deploy/llm-client/` in the repository runs this client in a container together with a llama.cpp server of its own, so it works without the chat app: `node execute.mjs up llm-client`, then `node execute.mjs llm-client load qwen3.5-4b` and `node execute.mjs llm-client ask "Why is the sky blue?"`. See `deploy/llm-client/README.md`.

## Check which model is loaded

```powershell
python -m tiny_agent.components.client.check                  # every model and its status
python -m tiny_agent.components.client.check gemma4-e4b       # exit code 0 if loaded, 1 if not, 2 if the server is down
```

## Settings from a `.env` file

`env.load_env()` reads `.env` (or the file named by `TINYAGENT_ENV_FILE`) and sets each name that the real environment has not already set. A missing, empty or unreadable file is fine, and values are never printed or logged. It understands comments, `export NAME=value`, double quotes (with `\n`, `\t`, `\"`, `\\`), single quotes (literal) and trailing ` # comments`.

```python
from tiny_agent.config import env

env.load_env()   # then os.environ["TINYAGENT_LLM_URL"] etc. are available
```

`python -m tiny_agent.components.client.check` uses `TINYAGENT_LLM_URL` as its default server. The names are listed in `tiny_agent/.env.example`. `LLM` takes its settings only as arguments, so pass what you read from the environment yourself.

## Tests

```powershell
python -m pytest tiny_agent/tests/test_client.py
```

The tests live with the rest of the package's tests in `tiny_agent/tests/` and use its `conftest.py` (the fake model server).

From the repository root, `node execute.mjs test llm` runs the same tests.
