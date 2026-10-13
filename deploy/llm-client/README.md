# deploy/llm-client

The LLM client (`code/llm_client`) and its own model server, deployed on their own. No chat app and no other folder in `deploy/` is needed.

| Service | What it is |
|---|---|
| `llama` | llama.cpp in router mode serving the models in `models.ini` (a copy of the chat app's). Published on <http://localhost:3200/v1>, this machine only, no API key. |
| `llm-client` | The client in a container, run on demand. |

```
cd deploy/llm-client
docker compose up -d --build                                    # start the model server
docker compose run --rm llm-client                              # which models are loaded
docker compose run --rm llm-client python load.py qwen3.5-4b    # load a model and wait (first time: downloads it)
docker compose run --rm llm-client python ask.py "Why is the sky blue?"   # the model answers
docker compose down
```

From the repo root: `node execute.mjs up llm-client`, `llm-client`, `llm-client load qwen3.5-4b`, `llm-client ask "Why is the sky blue?"` (add `--model NAME` before the question to pick a model) and `down llm-client`.

## Using it from the rest of the repo

- **The experimentation scripts** (`experimentation/illustrated_agents`) talk to `http://localhost:3200/v1` with the model names in `models.ini`, so with the model server up they run as they are: `.venv/Scripts/python experimentation/trigger_ias.py` (a popup lists them). The first request for a model loads it, which takes a while the first time.
- **The tiny agent** (`deploy/tiny-agent`, or the `tiny-agent` command) reaches the same server through `host.docker.internal:3200`, its default, so `node execute.mjs tiny-agent` chats with the model this folder serves.
- The server uses the **same port and the same GPU as the chat app**, so run this folder or the chat app, not both. Change the port with `LLAMA_PORT`.

## Settings and files

Copy `.env.example` to `.env` to change the port, the models folder or the server the client asks. To reuse the models the chat app already downloaded instead of downloading them again, set `MODELS_CACHE=../../experimentation/chat-app/models/cache`. It needs an NVIDIA GPU that Docker can see.

| File | What it is |
|---|---|
| `docker-compose.yaml` | The two services. Project name `tiny-agent-llm-client`. |
| `models.ini` | The models llama.cpp serves, a copy of `experimentation/chat-app/models/models.ini`. Keep the two in step. |
| `Dockerfile`, `Dockerfile.dockerignore` | The client image: `code/llm_client` plus `ask.py` and `load.py`, nothing installed. |
| `ask.py`, `load.py` | Send one question to a model; load a model and wait. |
| `.env.example` | The optional settings. |
