# Chat app

A small local chat app for trying TinyAgent's models, run with one `docker compose up`. A browser frontend talks to a backend, which talks to a llama.cpp server over its OpenAI-compatible API. llama.cpp downloads the models from Hugging Face the first time it loads them; there is no Ollama.

```
browser -> gateway (nginx, the only published port, :3200)
            |-- /          -> frontend (static React app)
            `-- /api, /v1  -> backend (FastAPI) -> llama.cpp router
```

```
experimentation/chat-app/
├── frontend/             React + Vite + Tailwind chat page (see frontend/README.md)
├── backend/              FastAPI service in front of llama.cpp
│   ├── server.py         API: /health, /api/models, /api/models/load, streamed /v1/*
│   ├── llama_client.py   httpx client for llama.cpp's router API
│   └── tests/            pytest suite with a fake router
├── models/               llama.cpp setup: model list and per-model settings
│   ├── models.ini        Model names -> Hugging Face repos, plus ctx-size and GPU layers
│   ├── cache/            The downloaded models (git-ignored, never removed by prune)
│   └── README.md         How to add or change a model
└── deploy/               Everything needed to run it in Docker
    ├── docker-compose.yaml
    ├── backend.Dockerfile, frontend.Dockerfile
    ├── gateway.nginx.conf    Single entry point (routes / and /api, /v1)
    ├── frontend.nginx.conf   Static file server for the React build
    └── .env.example      Optional settings (ports, models folder)
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

`backend/server.py` is the only thing that talks to llama.cpp, through its HTTP API (`LLAMA_URL`, default `http://llama:8080`). It uses only the standard library.

| Endpoint | What it does |
|---|---|
| `GET /health` | Backend status and whether llama.cpp is up |
| `GET /api/models` | Model names from llama.cpp's `/v1/models` |
| `POST /api/chat` | `{"model", "messages"}` in, first choice out (`llama_client.chat`) |
| `/v1/*` | Streamed straight through to llama.cpp; the frontend uses this for live tokens and tool calls |

```powershell
curl http://localhost:3200/api/models
curl http://localhost:3200/api/chat -d '{"model":"gemma4-e4b","messages":[{"role":"user","content":"Hi!"}]}'
```

## llama.cpp server

`experimentation/docker-compose.yaml` runs `llama-server` (CUDA build) in router mode. It downloads each model from Hugging Face the first time the model is loaded and keeps it in `chat-app/models/cache`; `models.ini` maps the names clients use to the Hugging Face repos.

```powershell
cd experimentation
docker compose up -d --build
docker compose logs -f llama
```

Then open the app at <http://localhost:3200>.

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

## Status

The whole stack runs end to end with `node execute.mjs up`, and `node execute.mjs test` runs the tests. llama.cpp's `/v1/models` `status` field (used to show which models are loaded) is read if present and ignored otherwise.
