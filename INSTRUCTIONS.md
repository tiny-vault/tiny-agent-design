# Instructions

How to set up and run everything in this repository. All commands are PowerShell, run from the repo root.

## 1. TinyAgent (the skeleton)

No runtime dependencies, Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python main.py
```

Expected output: `Received: Hello, agent!`

## 2. Documentation

```powershell
pip install -r docs/requirements.txt
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`.

## 3. Chat app (experiment)

Lives in `experimentation/chat-app/`. Full details are in its
[README](experimentation/chat-app/README.md).

### Frontend

Needs Node 20+ and [Ollama](https://ollama.com) running on port 11434 with a model pulled
(for example `gemma4:e4b-it-qat`).

```powershell
cd experimentation/chat-app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. `npm run build` makes a production bundle in `dist/`.

### Backend CLI

Sends one message to Ollama.

```powershell
python experimentation/chat-app/backend/chat_ollama.py gemma4:e4b-it-qat "Hi! How's life?"
```

### llama.cpp server (optional, not yet used by the frontend)

Needs Podman or Docker with NVIDIA GPU support.

```powershell
cd experimentation/chat-app
docker compose up -d
docker compose logs -f llama
```

Models that are not installed yet are downloaded first (set `PULL_MODELS` to
change the list). The API is served at <http://localhost:8080/v1>.

## Layout

```
tiny-agent-design/
├── main.py            # entry point
├── tiny_agent.py      # the TinyAgent harness
├── docs/              # Sphinx documentation
└── experimentation/
    └── chat-app/      # frontend, backend CLI, llama.cpp setup
```
