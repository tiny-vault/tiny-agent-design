# Chat app

A small local chat app for trying TinyAgent's models, started with `node execute.mjs up` (its compose file and Dockerfiles are in `deploy/chat-app/`). A browser frontend talks to a backend, which talks to a llama.cpp server over its OpenAI-compatible API. llama.cpp downloads the models from Hugging Face the first time it loads them; there is no Ollama.

```
browser -> gateway (nginx, the only published port, :3200)
            |-- /          -> frontend (static React app)
            `-- /api, /v1  -> backend (FastAPI) -> llama.cpp router
```

```
experimentation/
└── chat-app/             The app's source and state; its Docker files are in deploy/chat-app/
    ├── frontend/         React + Vite + Tailwind chat page (see frontend/README.md)
    ├── backend/          FastAPI service in front of llama.cpp
    │   ├── server.py     API: /health, /api/models, /api/models/load, streamed /v1/*
    │   ├── llama_client.py   httpx client for llama.cpp's router API
    │   └── tests/        pytest suite with a fake router
    ├── models/           llama.cpp setup: model list and per-model settings
    │   ├── models.ini    Model names -> Hugging Face repos, plus ctx-size and GPU layers
    │   ├── cache/        The downloaded models (git-ignored, never removed by prune)
    │   └── README.md     How to add or change a model
    ├── logs/             Generated, git-ignored: cli.log, main.log and OpenObserve's data
    └── secrets/          Generated, git-ignored: the Docker secrets (random values)
```

## Frontend

A single page with tool calling, a step trace, citations and streaming. It calls `/v1/...` on the same origin; nginx (in Docker) or the Vite dev server forwards that to the backend.

```powershell
cd experimentation/chat-app/frontend
npm install
npm run dev
```

Open <http://localhost:5173> (needs the stack running; the dev server forwards `/api` and `/v1` to the gateway on port 3200). Details are in [frontend/README.md](frontend/README.md).

## Backend

`backend/server.py` (FastAPI and httpx) is the only thing that talks to llama.cpp, through its HTTP API (`LLAMA_URL`, default `http://llama:8080`). It sends llama.cpp's API key, which it reads from a Docker secret, on every call; llama.cpp refuses callers without it.

| Endpoint | What it does |
|---|---|
| `GET /health` | Backend status and whether llama.cpp is up |
| `GET /api/models` | `[{id, status}]`, status `loaded`, `loading` or `unloaded`. Only the models in `models.ini`: the router also lists each cached download as an extra entry, which is left out |
| `POST /api/models/load` | Load a model through the router and wait until it is loaded (allows 30 minutes, because the first load downloads it). 404 unknown, 500 failed, 504 timed out. One switch at a time; it retries while llama.cpp is still busy |
| `POST /api/chat` | `{"model", "messages"}` in, first choice out |
| `GET /v1/models` | The same model list in OpenAI format |
| `POST /v1/chat/completions` | Streamed to llama.cpp, but refused with 409 unless the model is already loaded, so a chat never triggers a surprise switch |
| other `/v1/*` | Streamed straight through; the browser's own `Authorization` header is never forwarded |

Every error body is `{"error": {"message", "code"}}`.

```powershell
curl http://localhost:3200/api/models
curl http://localhost:3200/api/chat -d '{"model":"gemma4-e4b","messages":[{"role":"user","content":"Hi!"}]}'
```

## llama.cpp server

`deploy/chat-app/docker-compose.yaml` runs `llama-server` (CUDA build) in router mode. It downloads each model from Hugging Face the first time the model is loaded and keeps it in `chat-app/models/cache`; `models.ini` maps the names clients use to the Hugging Face repos.

```powershell
node execute.mjs up
node execute.mjs logs llama
```

Then open the app at <http://localhost:3200>. The model server of `deploy/llm-client` uses the same port and GPU, so run one of them at a time.

The server has an OpenAI-compatible API on port 8080 inside the compose network. It is not published to the host and its built-in web page is disabled (`LLAMA_ARG_WEBUI=0`), so the custom frontend is the only UI. From your machine, use the gateway, which forwards `/v1/*` to the backend: <http://localhost:3200/v1/models>.

### Models are downloaded from Hugging Face

The first time a model is loaded, llama.cpp downloads it from Hugging Face into `chat-app/models/cache` (several GB, so the first load takes minutes; the app shows "Loading..." until it is done). Later loads and restarts use the cached file, and nothing else in the stack downloads models. To fetch them ahead of time:

```powershell
node execute.mjs download            # every model in models.ini
node execute.mjs download gemma4-e4b # just one
```

| Name in requests | Hugging Face source | Size |
|---|---|---|
| `gemma4-e4b` | `google/gemma-4-E4B-it-qat-q4_0-gguf` (Google's quantisation-aware 4-bit build) | about 5 GB |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF`, `Q4_K_M` | about 2.7 GB |

To add a model, add a section with `hf-repo` to `models.ini`; see [models/README.md](models/README.md).

### Switching models at runtime

Send a different `model` in the request. The server loads it on demand and unloads the previous one.

```powershell
curl http://localhost:3200/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hi!"}]}'
```

`GET /v1/models` lists the available names.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_MODELS_MAX` | `1` | Models kept loaded at once |
| `LLAMA_LOAD_TIMEOUT` | `1800` | Seconds the backend waits for a model to load (the first load includes the download) |

Per-model settings (`ctx-size`, `n-gpu-layers`, `jinja`) live in `models.ini`.

### GPU

The compose file reserves the NVIDIA GPU and `models.ini` offloads all layers (`n-gpu-layers = 99`). The development laptop has 6 GB of VRAM, so a large model plus its context cache may not fit. If it runs out of memory, lower `n-gpu-layers` in `models.ini`; the remaining layers run on the CPU.

## Measured results

Measured on the development laptop (RTX 3060 Laptop, 6 GB of VRAM).

| What | Result |
|---|---|
| Switching to `qwen3.5-4b`, models on disk | 35 s |
| Switching to `gemma4-e4b`, models on disk | 61 s and 69 s |
| First load including the Hugging Face download | `qwen3.5-4b` 6 min (3.4 GB), `gemma4-e4b` 15 to 20 min (5.9 GB) |
| A chat answer starts, model already loaded | about 1 s |
| GPU memory with `gemma4-e4b` loaded | 5.3 of 6 GB, so both models cannot be loaded together |
| Disk used by the two downloaded models | 9.3 GB in `models/cache` (the old Ollama copies, 9.5 GB, were deleted) |

Every switch is a full unload and reload because only one model fits. See `node execute.mjs test` for the tests: backend 60, model client 123, `tiny_agent` 107 and the stub-based browser tests 35 pass.

## Status

The whole stack runs end to end with `node execute.mjs up`, including the log pipeline (OpenObserve and Vector) and the Docker secrets. llama.cpp downloads the models from Hugging Face itself; nothing else is needed. The `status` field llama.cpp reports in `/v1/models` is what the app uses to know which model is loaded.
