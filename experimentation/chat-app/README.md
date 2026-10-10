# Chat app

A small local chat app for trying TinyAgent's models, run with one `docker compose up`. A browser frontend talks to a backend, which talks to a llama.cpp server over its OpenAI-compatible API. The models are the ones Ollama already downloaded (missing ones are fetched).

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
├── models/               llama.cpp setup: model list, per-model settings, downloader
│   ├── models.ini        Model names -> Ollama blob files, plus ctx-size and GPU layers
│   ├── pull-models.sh    Downloads missing models (run by compose)
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

`deploy/docker-compose.yaml` runs `llama-server` (CUDA build) in router mode. Ollama stores each model as a plain GGUF file named `sha256-<digest>`, so the compose file mounts your Ollama `models/blobs` folder read-only and `models.ini` maps friendly names to those files. Nothing is copied, and Ollama is unaffected.

```powershell
cd experimentation/chat-app/deploy
docker compose up -d --build
docker compose logs -f llama
```

Then open the app at <http://localhost:3200>.

The server has an OpenAI-compatible API on port 8080 inside the compose network. It is not published to the host and its built-in web page is disabled (`LLAMA_ARG_WEBUI=0`), so the custom frontend is the only UI. From your machine, use the gateway, which forwards `/v1/*` to the backend: <http://localhost:3200/v1/models>.

### Missing models are downloaded automatically

`docker compose up` first runs a one-shot `pull-models` job (the `ollama/ollama` image running `pull-models.sh`). For each name in `PULL_MODELS` it checks whether Ollama already has the model, downloads it if not, and exits. Only then does `llama` start. Models you already have are skipped, so a second start needs no network.

```powershell
# Different or extra models (space separated Ollama names):
$env:PULL_MODELS = "gemma4:e4b-it-qat qwen3.5:4b llama3.2:3b"
docker compose up -d
```

Downloads go into `OLLAMA_MODELS_DIR`, the same folder the desktop Ollama uses, so both share one copy. The first download is several GB; follow it with `docker compose logs -f pull-models`.

After adding a model to `PULL_MODELS`, also add a section for it in `models.ini` (see below), or llama-server will not know the name.

`models.ini` points at blob files by digest. If the registry publishes a new build of a tag, the downloaded digest differs and the path in `models.ini` must be updated to match.

### Models

| Name in requests | Ollama model |
|---|---|
| `gemma4-e4b` | `gemma4:e4b-it-qat` |
| `qwen3.5-4b` | `qwen3.5:4b` |

To add one, read its digest from `~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>` (the layer with media type `...image.model`) and add a section to `models.ini`.

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
| `OLLAMA_MODELS_DIR` | `%USERPROFILE%/.ollama/models` (`$HOME` on Linux/macOS) | Ollama model folder to mount and download into |
| `PULL_MODELS` | `gemma4:e4b-it-qat qwen3.5:4b` | Models downloaded when missing |

Per-model settings (`ctx-size`, `n-gpu-layers`, `jinja`) live in `models.ini`.

### GPU

The compose file reserves the NVIDIA GPU and `models.ini` offloads all layers (`n-gpu-layers = 99`). The development laptop has 6 GB of VRAM, so a large model plus its context cache may not fit. If it runs out of memory, lower `n-gpu-layers` in `models.ini`; the remaining layers run on the CPU.

## Status

The frontend, backend and llama.cpp wiring is written and the frontend builds. The full `docker compose up` stack has not been run end to end yet. llama.cpp's `/v1/models` `status` field (used to show which models are loaded) is read if present and ignored otherwise.
