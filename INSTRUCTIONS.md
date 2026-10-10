# Instructions

How to set up and run everything in this repository. All commands are PowerShell, run from the repo root.

## 1. TinyAgent (the skeleton)

No runtime dependencies, Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m tiny_agent.main
```

Expected output: `Received: Hello, agent!`

## 2. Documentation

```powershell
pip install -r docs/requirements.txt
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`.

## 3. Chat app (experiment)

Lives in `experimentation/chat-app/`. Full details are in its [README](experimentation/chat-app/README.md).

### Everything in Docker (frontend + backend + llama.cpp)

Needs Docker with NVIDIA GPU support. Models that are not installed yet are downloaded first (set `PULL_MODELS` to change the list).

```powershell
cd experimentation/chat-app/deploy
docker compose up -d --build
docker compose logs -f llama
```

| Service | URL |
|---|---|
| App, API and health (the gateway, the only published port) | <http://localhost:3200> |
| Backend health | <http://localhost:3200/health> |
| llama.cpp API (via the backend) | <http://localhost:3200/v1/models> |

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
