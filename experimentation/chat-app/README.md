# Chat app

A small local chat app for trying TinyAgent's models. It has a browser
frontend, a one-shot command-line client, and a llama.cpp server that reuses
the models Ollama already downloaded.

```
experimentation/chat-app/
├── frontend/             React + Vite + Tailwind chat page (see frontend/README.md)
├── backend/
│   └── chat_ollama.py    One-message CLI client for Ollama
├── docker-compose.yaml   Model downloader + llama.cpp server (CUDA, router mode)
├── pull-models.sh        Downloads missing models (run by compose)
└── models.ini            Model names -> Ollama blob files
```

## Frontend

A single page with tool calling, a step trace, citations and streaming. It
currently talks to **Ollama** on port 11434.

```powershell
cd experimentation/chat-app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Details are in [frontend/README.md](frontend/README.md).

## Backend CLI

Sends one message to Ollama's OpenAI-compatible endpoint, using only the
standard library.

```powershell
python experimentation/chat-app/backend/chat_ollama.py gemma4:e4b-it-qat "Hi! How's life?"
```

## llama.cpp server

`docker-compose.yaml` runs `llama-server` (CUDA build) in router mode. Ollama
stores each model as a plain GGUF file named `sha256-<digest>`, so the compose
file mounts `~/.ollama/models/blobs` read-only and `models.ini` maps friendly
names to those files. Nothing is copied, and Ollama is unaffected.

```powershell
cd experimentation/chat-app
docker compose up -d
docker compose logs -f llama
```

The server listens on <http://localhost:8080> with an OpenAI-compatible API.

### Missing models are downloaded automatically

`docker compose up` first runs a one-shot `pull-models` job (the `ollama/ollama`
image running `pull-models.sh`). For each name in `PULL_MODELS` it checks
whether Ollama already has the model, downloads it if not, and exits. Only then
does `llama` start. Models you already have are skipped, so a second start
needs no network.

```powershell
# Different or extra models (space separated Ollama names):
$env:PULL_MODELS = "gemma4:e4b-it-qat qwen3.5:4b llama3.2:3b"
docker compose up -d
```

Downloads go into `OLLAMA_MODELS_DIR`, the same folder the desktop Ollama uses,
so both share one copy. The first download is several GB; follow it with
`docker compose logs -f pull-models`.

After adding a model to `PULL_MODELS`, also add a section for it in
`models.ini` (see below), or llama-server will not know the name.

`models.ini` points at blob files by digest. If the registry publishes a new
build of a tag, the downloaded digest differs and the path in `models.ini` must
be updated to match.

### Models

| Name in requests | Ollama model |
|---|---|
| `gemma4-e4b` | `gemma4:e4b-it-qat` |
| `qwen3.5-4b` | `qwen3.5:4b` |

To add one, read its digest from
`~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>` (the layer
with media type `...image.model`) and add a section to `models.ini`.

### Switching models at runtime

Send a different `model` in the request. The server loads it on demand and
unloads the previous one.

```powershell
curl http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hi!"}]}'
```

`GET /v1/models` lists the available names.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_MODELS_MAX` | `1` | Models kept loaded at once |
| `OLLAMA_MODELS_DIR` | `~/.ollama/models` | Ollama model folder to mount and download into |
| `PULL_MODELS` | `gemma4:e4b-it-qat qwen3.5:4b` | Models downloaded when missing |

Per-model settings (`ctx-size`, `n-gpu-layers`, `jinja`) live in `models.ini`.

### GPU

The compose file reserves the NVIDIA GPU and `models.ini` offloads all layers
(`n-gpu-layers = 99`). The development laptop has 6 GB of VRAM, so a large
model plus its context cache may not fit. If it runs out of memory, lower
`n-gpu-layers` in `models.ini`; the remaining layers run on the CPU.

## Status

The llama.cpp setup has not been run yet, and the frontend does not use it:
it still calls Ollama's API. Switching the frontend to
`http://localhost:8080/v1` is still to do.
