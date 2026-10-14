# Instructions

How to set up and run everything in this repository. All commands are PowerShell, run from the repo root.

## 1. TinyAgent (the harness, the LLM client and the terminal chat)

No runtime dependencies, Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-Content .venv\Lib\site-packages\sitecustomize.py "import sys; sys.dont_write_bytecode = True"   # optional: no __pycache__ folders in the repo
python -m tiny_agent.main
```

The demo tells the agent the authors' names and then asks for them in a second run on the same agent, so memory is what answers; it prints the second reply. It needs a running model server with a model loaded: the chat app (`node execute.mjs up`, section 3) or only the llm client's model server (`node execute.mjs up llm-client`, section 4). It reads the same `TINYAGENT_MODEL` (default `gemma4-e4b`), `TINYAGENT_LLM_URL` and `TINYAGENT_API_KEY_FILE` settings as the `tiny-agent` command below. Neither server needs a key from you: the chat app's backend holds the llama.cpp key, and the llm client's server listens on this machine only.

### Chat from the terminal

Install the project once and you get a `tiny-agent` command (needs a model server with a model loaded: the chat app from section 3 or the llm client's server from section 4):

```powershell
python -m pip install -e .
tiny-agent                          # interactive chat
tiny-agent "Why is the sky blue?"   # one question, one answer, then exit
tiny-agent --demo                   # run the main.py demo first, then chat
tiny-agent --model qwen3.5-4b --think --system "Be brief."
```

It chats with the model that is loaded right now unless you pass `--model`. In the chat, `/help` lists the commands: `/models`, `/model NAME`, `/think on|off`, `/clear`, `/exit`. Without installing, `python -m tiny_agent` does the same. Settings can also come from `TINYAGENT_MODEL` and `TINYAGENT_LLM_URL`.

## 2. Documentation

```powershell
pip install -r docs/requirements.txt
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

To read them on a local web server, run `node execute.mjs docs`: it builds them into a Docker image (a documentation warning fails the build) and serves them at <http://localhost:3300> (this machine only; `node execute.mjs down docs` stops them, `DOCS_PORT` changes the port). Without Docker, `node execute.mjs docs --local` builds with the Sphinx above and serves the result (Ctrl-C stops it).

Open `docs/_build/html/index.html`.

## 3. Chat app (experiment)

Lives in `experimentation/chat-app/`. Full details are in its [README](experimentation/chat-app/README.md).

### Everything in Docker (frontend + backend + llama.cpp)

Needs Docker with NVIDIA GPU support. llama.cpp downloads the models from Hugging Face the first time they are loaded (several GB; `node execute.mjs download` fetches them ahead of time) and keeps them in `experimentation/chat-app/models/cache`. There is no Ollama.

```powershell
node execute.mjs up                # builds and starts it, then waits until it answers
node execute.mjs logs llama        # follow the model server's log
```

The compose file is `deploy/chat-app/docker-compose.yaml`. By hand: `node deploy/chat-app/init-secrets.mjs` once (creates the secret files), then `docker compose up -d --build` from `deploy/chat-app`. Optional settings (`GATEWAY_PORT`, `OPENOBSERVE_PORT`, `LLAMA_MODELS_MAX`) go in `deploy/chat-app/.env`; see `.env.example` there.

| Service | URL |
|---|---|
| App, API and health (the gateway, the only published port) | <http://localhost:3200> |
| Backend health | <http://localhost:3200/health> |
| llama.cpp API (via the backend) | <http://localhost:3200/v1/models> |
| Logs, searchable (OpenObserve, this machine only) | <http://localhost:5080> |

### Logs, secrets and cleaning up

`node execute.mjs up` starts the complete stack, including the log pipeline: OpenObserve stores and searches the logs and Vector collects them. Everything is in one place, <http://localhost:5080>, which opens already logged in (no login screen):

* stream `docker`: the chat-app containers (gateway, frontend, backend, llama.cpp), field `service` tells them apart;
* stream `tiny_agent`: `tiny-agent` (service `tiny-agent-cli`) and `python -m tiny_agent.main` (service `tiny-agent-main`), which also write `experimentation/chat-app/logs/cli.log` and `main.log`.

```powershell
node execute.mjs observe              # URL and what is collected
node execute.mjs observe --password   # also print the login, if you ever need it
```

The OpenObserve login and the llama.cpp API key are Docker secrets: `up` creates them with random values in `experimentation/chat-app/secrets/` (not in git). The model server only accepts calls that carry the key, and only the backend has it.

```powershell
node execute.mjs prune --dry-run   # show what would be removed
node execute.mjs prune             # remove the whole setup, asking first
```

`prune` removes the containers, networks, every volume of the stack, the secrets, the logs (including OpenObserve's data) and the build output. It does **not** touch your downloaded models, the Docker images (so the next `up` needs no big downloads), source, `.env`, `.venv` or git.

### How long things take

Measured on the development laptop (6 GB GPU), with the models already on disk unless stated.

| What | Time |
|---|---|
| First `node execute.mjs up` (downloads images, builds the app) | several minutes; the llama.cpp image alone is 7 GB |
| First load of a model, which downloads it from Hugging Face | `qwen3.5-4b` (3.4 GB) 6 minutes; `gemma4-e4b` (5.9 GB) 15 to 20 minutes |
| Switching to `qwen3.5-4b` | 35 s |
| Switching to `gemma4-e4b` | 61 to 69 s |
| A chat answer starts, model already loaded | about 1 s |
| `node execute.mjs test backend`, `llm`, `agent`, `mocked` | 14 s, 16 s, 11 s, about 50 s |

Only one model fits in GPU memory (`gemma4-e4b` alone uses 5.3 of 6 GB), so every switch is a full unload and reload. Fetch both models ahead of time with `node execute.mjs download`, which also shows the progress.

### Frontend dev server

Needs Node 20+ and the stack running (the dev server forwards the API to the gateway on port 3200).

```powershell
cd experimentation/chat-app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. `npm run build` makes a bundle in `dist/`.

## 4. The other Docker parts, each on its own

The Docker setup is four independent folders in `deploy/`, each with its own compose file, README and `.env.example`. Start only what you need; see [deploy/README.md](deploy/README.md).

| Need | Command | Reach it at |
|---|---|---|
| A model to talk to, without the chat app (run the `experimentation/illustrated_agents` scripts, `tiny-agent`) | `node execute.mjs up llm-client`, then `node execute.mjs llm-client load qwen3.5-4b` | <http://localhost:3200/v1> |
| Ask that model something, from the llm client | `node execute.mjs llm-client ask "Why is the sky blue?"` | the terminal |
| The tiny agent in a container | `node execute.mjs tiny-agent "Why is the sky blue?"` | the terminal |
| The Sphinx docs | `node execute.mjs up docs` (or `node execute.mjs docs`) | <http://localhost:3300> |

Stop a part with `node execute.mjs down <chat-app|llm-client|tiny-agent|docs|all>`. The chat app and the llm client's model server both use port 3200 and the GPU, so run one of them at a time. The llm client reuses the models the chat app downloaded when `deploy/llm-client/.env` sets `MODELS_CACHE=../../experimentation/chat-app/models/cache` (see its `.env.example`). If a model container fails with `WSL environment detected but no adapters were found`, Docker lost the GPU: `wsl --shutdown`, restart Docker Desktop, and check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L`.

## Markdown style

Markdown is kept unwrapped (one paragraph per line). Enable the commit hook that enforces it once per clone with `git config core.hooksPath .githooks`, and check everything with `node tools/unwrap-md.mjs --check`.

## Layout

```
tiny-agent-design/
├── tiny_agent/      # the TinyAgent package, self-contained
│   ├── main.py      #   the two-turn memory demo
│   ├── cli.py       #   the tiny-agent terminal chat
│   ├── agent.py, step.py, trajectory.py, response.py  #   the harness (TinyAgent) and what it records
│   ├── memory/, tools/, client/  #   the memories, the tools and the model client, each with its base class
│   ├── config/     #   the .env loader
├── deploy/            # Docker: chat-app/, llm-client/, tiny-agent/, docs/ (a compose file each)
├── docs/              # Sphinx documentation
├── execute.mjs        # the one entry point for Docker and tests
└── experimentation/
    ├── chat-app/      # frontend, backend, models, logs, secrets
    ├── illustrated_agents/  # runnable book scripts, one file per concept
    └── trigger_ias.py # popup to run them
```
