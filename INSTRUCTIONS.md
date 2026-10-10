# Instructions

How to set up and run everything in this repository. All commands are PowerShell, run from the repo root.

## 1. TinyAgent (the skeleton)

No runtime dependencies, Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m tiny_agent.main
```

Expected output: `Received: Hello, agent!`. This demo does not call a model; to chat with one, use the `tiny-agent` command below.

### Chat from the terminal

Install the project once and you get a `tiny-agent` command (needs the chat app stack from section 3 running with a model loaded):

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

Open `docs/_build/html/index.html`.

## 3. Chat app (experiment)

Lives in `experimentation/chat-app/`. Full details are in its [README](experimentation/chat-app/README.md).

### Everything in Docker (frontend + backend + llama.cpp)

Needs Docker with NVIDIA GPU support. llama.cpp downloads the models from Hugging Face the first time they are loaded (several GB; `node execute.mjs download` fetches them ahead of time) and keeps them in `experimentation/chat-app/models/cache`. There is no Ollama.

```powershell
cd experimentation
docker compose up -d --build
docker compose logs -f llama
```

| Service | URL |
|---|---|
| App, API and health (the gateway, the only published port) | <http://localhost:3200> |
| Backend health | <http://localhost:3200/health> |
| llama.cpp API (via the backend) | <http://localhost:3200/v1/models> |
| Logs, searchable (OpenObserve, this machine only) | <http://localhost:5080> |

### Logs, secrets and cleaning up

`node execute.mjs up` starts the complete stack, including the log pipeline: OpenObserve stores and searches the logs and Vector collects them. Everything is in one place, <http://localhost:5080>, which opens already logged in (no login screen):

* stream `docker`: the chat-app containers (gateway, frontend, backend, llama.cpp), field `service` tells them apart;
* stream `tiny_agent`: `tiny-agent` (service `tiny-agent-cli`) and `python -m tiny_agent.main` (service `tiny-agent-main`), which also write `experimentation/logs/cli.log` and `main.log`.

```powershell
node execute.mjs observe              # URL and what is collected
node execute.mjs observe --password   # also print the login, if you ever need it
```

The OpenObserve login and the llama.cpp API key are Docker secrets: `up` creates them with random values in `experimentation/secrets/` (not in git). The model server only accepts calls that carry the key, and only the backend has it.

```powershell
node execute.mjs prune --dry-run   # show what would be removed
node execute.mjs prune             # remove the whole setup, asking first
```

`prune` removes the containers, networks, every volume of the stack, the secrets, the logs (including OpenObserve's data) and the build output. It does **not** touch your downloaded models, the Docker images (so the next `up` needs no big downloads), source, `.env`, `.venv` or git.

### Frontend dev server

Needs Node 20+ and the stack running (the dev server forwards the API to the gateway on port 3200).

```powershell
cd experimentation/chat-app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. `npm run build` makes a bundle in `dist/`.

## Markdown style

Markdown is kept unwrapped (one paragraph per line). Enable the commit hook that enforces it once per clone with `git config core.hooksPath .githooks`, and check everything with `node tools/unwrap-md.mjs --check`.

## Layout

```
tiny-agent-design/
├── tiny_agent/
│   ├── main.py        # entry point
│   └── shared/        # the TinyAgent harness, LLM client, model check
├── docs/              # Sphinx documentation
└── experimentation/
    └── chat-app/      # frontend, backend, llama.cpp compose
```
