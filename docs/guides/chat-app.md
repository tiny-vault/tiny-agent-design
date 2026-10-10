# Chat app

The chat app (in `experimentation/chat-app/`) is a local chat UI that talks to models served by llama.cpp, all in Docker. It is an experiment next to TinyAgent and does not use the `tiny_agent` package yet. One command starts the whole stack:

```powershell
node execute.mjs up
```

## How it fits together

```{mermaid}
flowchart LR
    B["Browser"] --> G["gateway (nginx, port 3200)"]
    G -->|"/"| F["frontend (static React app)"]
    G -->|"/api, /v1"| K["backend (FastAPI)"]
    K -->|"API key"| L["llama.cpp (router mode)"]
    L -->|"first load"| H["Hugging Face"]
    L --> C[("models/cache")]
    V["Vector"] -->|"logs"| O["OpenObserve (port 5080)"]
```

* The gateway is the only published port for the app. The backend is the only code that talks to llama.cpp, and it refuses to chat with a model that is not loaded (HTTP 409), so a chat can never trigger a surprise model switch.
* The frontend lets you chat only when the router reports the selected model as loaded. Switching freezes the page until the load finishes, and a failed load shows the reason and a Load button.
* llama.cpp runs in router mode: it knows several models but keeps one loaded at a time (`LLAMA_MODELS_MAX=1`).

## Models

llama.cpp downloads each model from Hugging Face the first time it is loaded and keeps it in `experimentation/chat-app/models/cache`. There is no Ollama and no separate download service.

| Name | Source | On disk |
|---|---|---|
| `gemma4-e4b` | `google/gemma-4-E4B-it-qat-q4_0-gguf` (Google's quantisation-aware 4-bit build) | 5.9 GB |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF`, `Q4_K_M` | 3.4 GB |

The sizes include a small vision file each repository carries. To add a model, add a section with `hf-repo` to `models.ini`. `node execute.mjs download` fetches models ahead of time and shows the progress.

## Measured results

Measured on the development laptop (6 GB GPU) while building this.

| What | Result |
|---|---|
| First load of `gemma4-e4b` (includes the download) | about 15 to 20 minutes |
| First load of `qwen3.5-4b` (includes the download) | 6 minutes |
| Switching to `qwen3.5-4b`, models already on disk | 35 s |
| Switching to `gemma4-e4b`, models already on disk | 61 s and 69 s |
| Answering with the model already loaded | about 1 s to start |
| GPU memory with `gemma4-e4b` loaded | 5.3 of 6 GB, so both models cannot be loaded together |

A switch is a full unload and reload because only one model fits. The time is mostly reading the file and copying it to the GPU, so the larger model takes longer. To make switching near-instant you would need a GPU with room for both models (then `LLAMA_MODELS_MAX=2`).

The backend allows 30 minutes for a load (`LLAMA_LOAD_TIMEOUT`) and the gateway 2000 seconds, because a first load includes a multi-gigabyte download.

## Logs

Everything is searchable in one place, OpenObserve at <http://localhost:5080>, which opens already logged in.

* Stream `docker`: the gateway, frontend, backend and llama.cpp containers (field `service`).
* Stream `tiny_agent`: the `tiny-agent` command (service `tiny-agent-cli`) and `python code/main.py` (service `tiny-agent-main`). They write `experimentation/chat-app/logs/cli.log` and `main.log`, and Vector ships those files to OpenObserve. The Python code itself never talks to OpenObserve.

```powershell
node execute.mjs observe              # the URL and what is collected
node execute.mjs observe --password   # also the login, if you ever need it
```

## Secrets

Credentials are Docker secrets: files in `experimentation/chat-app/secrets/` (not in git), created with random values by `node execute.mjs up`.

* The llama.cpp API key: llama.cpp requires it and only the backend has it. Without it a call gets 401. `/health` stays open.
* The OpenObserve login: OpenObserve cannot read secret files, so a small wrapper image reads them at start-up. Its password must have a lowercase letter, an uppercase letter, a digit and a special character, or it refuses to start; the generator always meets that rule.
* The OpenObserve page opens already logged in through a small nginx proxy. Anyone who can reach `127.0.0.1:5080` on the machine is the admin, while a bare API call with no login is still refused.

## Commands

All from the repository root.

| Command | What it does |
|---|---|
| `node execute.mjs up` | Start the complete stack and print the URLs |
| `node execute.mjs download [model]` | Fetch models from Hugging Face ahead of time |
| `node execute.mjs load <model>` | Load a model and wait until it is loaded |
| `node execute.mjs status`, `health`, `logs [service]` | See what is running |
| `node execute.mjs test [target]` | Run the tests |
| `node execute.mjs down` | Stop and remove the containers |
| `node execute.mjs prune` | Remove the whole setup except the downloaded models and the Docker images |

## Tests

| Target | What it runs | Result |
|---|---|---|
| `backend` | pytest against a fake llama.cpp router | 59 pass |
| `llm` | pytest for the `llm_client` package against a fake model server | 121 pass |
| `agent` | pytest for `tiny_agent` (chat command, logs, demo) against a fake model server | 93 pass |
| `mocked` | Playwright with the API stubbed in the browser: model-switch gating and chat edge cases | 35 pass |
| `live` | Playwright against the running stack: real model switching and OpenObserve | see `INSTRUCTIONS.md` for the latest run |

## Troubleshooting

* **OpenObserve exits at start-up:** its root password breaks the policy. `node execute.mjs up` replaces a weak password automatically.
* **`node execute.mjs load` fails after 5 minutes:** older versions used `fetch`, which gives up waiting for response headers after 5 minutes. The command now uses `node:http`.
* **A model switch is refused with "model limit reached":** llama.cpp is still finishing a generation you stopped. The backend retries until it clears.
* **Port 3200 or 5080 is taken:** set `GATEWAY_PORT` or `OPENOBSERVE_PORT` in `experimentation/.env`.
