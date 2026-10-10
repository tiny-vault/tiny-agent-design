# AGENTS.md

Guidance for AI coding agents (and people) working in this repository. Read this before changing anything; it lists what exists, how to run and test it, and the traps that have already cost time.

## Rules for `tiny_agent/`

The `tiny_agent/` folder has its own stricter rules, in `tiny_agent/AGENTS.md`: fix methods only, never rewrite or restructure, flag what is missing, and re-read a file before every edit. Read them before touching anything in that folder. They do not apply to the rest of the repo.

## What this repo is

Two things live here:

1. **TinyAgent** (`tiny_agent/`, `docs/`): a minimal, educational agent harness, built step by step. Currently a skeleton; no model is called.
2. **Chat app** (`experimentation/chat-app/`): a local chat UI that talks to models served by llama.cpp, all in Docker. It is an experiment and does not use `tiny_agent/` yet.

## Layout

```
tiny_agent/main.py             Demo entry point (python -m tiny_agent.main)
tiny_agent/cli.py              Terminal chat: the `tiny-agent` command (pip install -e .)
tiny_agent/shared/             TinyAgent harness, LLM client, model check, Response
pyproject.toml                 Packaging: the `tiny-agent` console command
tools/unwrap-md.mjs            Keeps Markdown unwrapped (used by the pre-commit hook)
docs/                          Sphinx docs (Furo theme)
INSTRUCTIONS.md                How to run everything
execute.mjs                    Control the Docker deployment and run the tests (start here)
experimentation/
├── docker-compose.yaml        The whole stack; .env, secrets/ and logs/ live next to it
├── observability/             OpenObserve image wrapper (reads secrets) and the Vector config
├── logs/ , secrets/           Generated, git-ignored: log files + OpenObserve data; Docker secrets
└── chat-app/
    ├── frontend/                  React 19 + Vite + Tailwind v4 (src/components, hooks, lib)
    ├── backend/                   FastAPI + httpx; tests/ has a fake llama router
    ├── models/                    llama.cpp things: models.ini (Hugging Face repos), cache/ (downloads), README
    └── deploy/                    Dockerfiles, nginx configs
```

## How the chat app fits together

```
browser -> gateway (nginx, the ONLY published port, 3200)
             |-- /          -> frontend (static files)
             `-- /api, /v1  -> backend (FastAPI) -> llama.cpp (router mode, internal)
```

- Nothing but the gateway is published to the host. Do not add `ports:` to other services; add routes to `experimentation/chat-app/deploy/gateway.nginx.conf` instead.
- The backend is the only code that talks to llama.cpp. It exposes `/api/models`, `/api/models/load`, `/api/chat`, a streamed `/v1/*` passthrough, and refuses `POST /v1/chat/completions` with **409** unless the model is already loaded.
- The frontend lets the user chat only when the router reports the selected model `loaded` and no switch is in progress. It never relies on llama.cpp loading a model implicitly.

## Commands

All from the repo root unless noted.

| Task | Command |
|---|---|
| Start everything | `node execute.mjs up` |
| Status, health, logs | `node execute.mjs status` / `health` / `logs [service]` |
| Switch model | `node execute.mjs load <model>` |
| Stop | `node execute.mjs down` |
| Run tests | `node execute.mjs test [backend\|mocked\|live\|e2e\|all]`, extra arguments after `--` go to the runner (see below) |
| Frontend dev | `cd frontend; npm install; npm run dev` (needs the stack up; proxies to the gateway) |
| Frontend build | `cd frontend; npm run build` |
| Docs | repo root: `sphinx-build -b html -n -W --keep-going docs docs/_build/html` |
| Demo agent | repo root: `python -m tiny_agent.main` |

Python lives in the repo's `.venv` (Python 3.12). Activate it, or call `.venv/Scripts/python` directly. Backend dependencies are in `backend/requirements*.txt`.

## Tests

`node execute.mjs test` is the one entry point; do not hand-roll the runners.

| Target | What it runs | Needs |
|---|---|---|
| `backend` | pytest in `backend/tests`, against a fake llama router | nothing |
| `agent` | pytest in `tiny_agent/tests` (check.py, llm.py, Response, TinyAgent, the demo) against a fake model server | nothing |
| `mocked` | Playwright `frontend/e2e/{gating,edge}.spec.js`, API stubbed in the browser, on a Vite dev server Playwright starts (so it tests the current source, not a Docker image) | nothing |
| `live` | Playwright `frontend/e2e/live.spec.js` through the gateway, real models | the stack up (`node execute.mjs up`) |
| `all` (default) | backend + mocked, plus live when the stack is up | |
| `e2e` | mocked + live | the stack up |
| `list` | list the tests without running them | |

Pass runner arguments after `--`, for example `node execute.mjs test mocked -- --grep Stop` or `node execute.mjs test backend -- -k load`. The command exits non-zero if any target fails and prints a PASS/FAIL summary.

When you change behaviour, add or update a test in the same change. The fake router (`backend/tests/fake_router.py`) and the browser stubs (`frontend/e2e/helpers.js`) copy how the real llama.cpp router behaves; if the real behaviour turns out to differ, fix the fake. The live tests rebuild nothing: after frontend or backend changes run `node execute.mjs rebuild` first.

## Conventions

- **Scripting on Windows:** for anything beyond a one-liner (bulk edits, generating files, multi-step logic) write a Node `.mjs` script and run it with `node file.mjs`. Shell heredocs and here-strings fail repeatedly here (quoting, `$` expansion, EOF errors). Keep throwaway scripts outside the repo.
- **Line endings:** `.gitattributes` normalises text to LF, and `*.sh` is forced to LF because those files run in Linux containers. Do not commit CRLF shell scripts.
- **Match the surrounding code:** docstrings and comments explain *why*, are short, and are written for a reader new to agents. Python is stdlib-light and typed where it helps; the frontend is plain JavaScript with function components and Tailwind utility classes (design tokens live in `frontend/src/index.css`).
- **Markdown is never hard-wrapped:** one paragraph or list item is one line, however long; the editor and browser wrap it on screen. Do not break prose at 80 columns. A pre-commit hook (`.githooks/pre-commit`) unwraps staged `.md` files, and `node tools/unwrap-md.mjs --check` verifies the whole repo (`--write` fixes it). After cloning, enable the hook once with `git config core.hooksPath .githooks`.
- **Docs move with code:** if you change a command, port, path or endpoint, update `INSTRUCTIONS.md` and the relevant README in the same change.
- **Errors from the backend** always look like `{"error": {"message", "code"}}`. Keep that shape; the frontend reads `message`.
- **Do not commit:** `.env`, model files, `node_modules`, `dist`, `.venv`, `docs/_build`. `.gitignore` already covers them; check `git status` before committing.

## Secrets and logs

* Credentials are **Docker secrets**: files in `experimentation/secrets/` (git-ignored), created with random values by `node execute.mjs up` and mounted at `/run/secrets/<name>`. Never put a password or key in the compose file, an environment variable, `.env`, an image or a log line. Today there are two: the OpenObserve login and `llama_api_key` (llama-server requires it; only the backend sends it).
* Services read a secret from a file: llama-server with `LLAMA_ARG_API_KEY_FILE`, the backend with `LLAMA_API_KEY_FILE`, Vector with `SECRET[docker.<name>]`. OpenObserve cannot read files, so `observability/openobserve.Dockerfile` adds a tiny shell whose entrypoint does it.
* The OpenObserve UI opens **already logged in**: the `observe` service (nginx, `observability/auto-login.sh` and `observe.nginx.conf.template`) builds the login cookie and browser state from the secrets and signs every browser in. OpenObserve has no setting for this, so it is a proxy. Trade-off: anyone who can reach `127.0.0.1:5080` on this machine is the admin; the API itself still refuses a call with no cookie or credentials. Keep redirects relative (`absolute_redirect off`): nginx listens on 80 inside the container.
* All logs go to OpenObserve (<http://localhost:5080>): container logs through Vector's Docker socket mount, and `tiny_agent/log.py` writes `experimentation/logs/cli.log` and `main.log` for `tiny-agent` and `main.py` only. Do not add file logging to the other modules; Docker already captures them.
* `node execute.mjs prune` removes the whole setup except the downloaded models and the Docker images (llama.cpp alone is several GB). It removes the stack's containers, networks and every volume labelled for the `chat-app` project, plus the generated secrets, logs and build output. It never runs `docker system prune`: this machine hosts other projects.

## Gotchas (learned the hard way)

- **llama.cpp router behaviour** (verified against the real server): `POST /models/load` returns `{"success":true}` immediately; the model then reports `loading` and later `loaded` in `GET /v1/models` (`status.value`). With `LLAMA_MODELS_MAX=1`, loading one model unloads the other. An unknown name gives 404. The built-in web page is disabled (`LLAMA_ARG_WEBUI=0`).
- **Model files:** llama.cpp downloads each model from Hugging Face the first time it is loaded (`hf-repo` in `models/models.ini`) into `models/cache/` (git-ignored; `prune` never removes it). There is no Ollama and no separate download service. A first load can take many minutes, so the backend allows `LLAMA_LOAD_TIMEOUT` (1800 s) and the gateway 2000 s; keep those in step if you change them.
- **Ports:** this machine already uses 3000, 8000, 8080 and 3001 for other projects. The gateway uses **3200** (`GATEWAY_PORT`). Check with `netstat` before picking a new one.
- **Compose project name** is pinned to `chat-app` (`name:` in the compose file) so moving files around does not create a second stack.
- **Compose does not expand `~`.** Use `USERPROFILE`/`HOME` or a full path.
- **Node on Windows:** calling `process.exit()` right after a `fetch` with a timeout can crash with a libuv assertion. Set `process.exitCode` instead.
- **GPU:** the dev laptop has 6 GB of VRAM. Large models plus context may not fit; lower `n-gpu-layers` in `models/models.ini`.
- **Browser automation** (Claude in Chrome) needs the extension connected; if it is not, fall back to API checks and Playwright.

## Before you finish a change

1. Run `node execute.mjs test` (backend + mocked, plus live when the stack is up).
2. If you touched the frontend, also run `npm run build` in `frontend/`.
3. If you touched Docker or nginx files, run `docker compose -f experimentation/docker-compose.yaml config -q`, then `node execute.mjs up` and `node execute.mjs health`.
4. Update the docs listed above, then check `git status` for anything that should not be committed.

## Attribution

TinyAgent follows *An Illustrated Guide to AI Agents* by Maarten Grootendorst and Jay Alammar (O'Reilly); the chat UI's look is adapted from Utopia by DeepLethe (Apache-2.0). See `README.md` and `experimentation/chat-app/frontend/README.md`.
