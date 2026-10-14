# AGENTS.md

Guidance for AI agents. `tiny_agent/` has a stricter `AGENTS.md` (fix methods only, never restructure, flag what is missing, re-read before every edit; it also covers `tiny_agent/components/client/` and `tiny_agent/config/`); read it before touching that folder.

**Never create, edit or delete anything in `experimentation/` unless the user explicitly asks for a change there.** A pasted snippet, a "merge into tiny_agent" or "test it" is not such a request: do the work in `tiny_agent/` (and run things from a scratchpad, not a new file here). This overrides every other rule in this file that says to create a script.

## What this is

1. **TinyAgent** (`tiny_agent/`, `docs/`): minimal educational agent harness. One model call per task, conversation in `Memory`; loop and planning are placeholders. Also the model client (`tiny_agent/components/client/`, `tiny_agent/config/`), the `tiny-agent` terminal chat (`cli.py`), demo `python -m tiny_agent.main`.
2. **Chat app** (`experimentation/chat-app/`): local chat UI for llama.cpp models, all in Docker; does not use `tiny_agent/`.

Key paths: `execute.mjs` (Docker + tests, start here), `INSTRUCTIONS.md` (how to run everything), `deploy/` (four independent parts, each with its own `docker-compose.yaml`, Dockerfiles, `.env.example`, README: `chat-app/` (llama.cpp, backend, frontend, gateway, `observability/`, `init-secrets.mjs`), `llm-client/`, `tiny-agent/` (both on demand), `docs/` (Sphinx)), `experimentation/chat-app/{frontend (React 19, Vite, Tailwind v4), backend (FastAPI), models (models.ini, cache/), logs/, secrets/}`.

## Architecture

```
browser -> gateway (nginx, ONLY published port, 3200)
             |-- /          -> frontend (static)
             `-- /api, /v1  -> backend (FastAPI) -> llama.cpp (router mode, internal)
```

- No `ports:` on other services; add routes in `deploy/chat-app/gateway.nginx.conf`.
- Only the backend talks to llama.cpp: `/api/models`, `/api/models/load`, `/api/chat`, streamed `/v1/*` passthrough; `POST /v1/chat/completions` is **409** unless the model is loaded.
- The frontend allows chat only when the selected model is `loaded` and no switch is in progress.
- Backend errors are always `{"error": {"message", "code"}}`; the frontend reads `message`.

## Commands (repo root)

| Task | Command |
|---|---|
| Start / stop | `node execute.mjs up [chat-app\|llm-client\|tiny-agent\|docs\|all]` / `down [target]` (default `chat-app`; each target is its own compose project in `deploy/<target>/`) |
| LLM client / tiny agent | `up llm-client` starts its own model server on `localhost:3200` (same port and GPU as the chat app: run one or the other; the `illustrated_agents` scripts run against it). Then `llm-client [model]` (loaded models) / `llm-client load <model>` / `llm-client ask "q"`, and `tiny-agent ["q"]`, which uses whatever answers on 3200 or `TINYAGENT_LLM_URL` |
| Status, health, logs | `status` / `health` / `logs [service]` |
| Switch model / rebuild | `load <model>` / `rebuild` |
| Notebook-cell scripts | `.venv/Scripts/python experimentation/trigger_ias.py [name]` (no name: popup REPL; `--list`; needs stack up, model loaded) |
| Tests | `test [backend\|agent\|llm\|mocked\|live\|e2e\|ias\|all\|list]`, runner args after `--` (e.g. `test backend -- -k load`) |
| Frontend | `cd experimentation/chat-app/frontend; npm run dev` (stack up) / `npm run build` |
| Docs | `docs` builds an image and serves <http://localhost:3300> (`DOCS_PORT`); `docs --local [--no-build] [--port N]` without Docker; build only: `sphinx-build -b html -n -W --keep-going docs docs/_build/html` |
| Educational audit | `/edu-audit <folder>` (local skill in `.claude/skills/edu-audit/`, git-ignored) |

All `node execute.mjs` commands except where shown. Python: `.venv/Scripts/python` (3.12). Backend deps: `backend/requirements*.txt`.

**No `__pycache__` in the repo:** `execute.mjs` and the Dockerfiles set `PYTHONDONTWRITEBYTECODE=1`; `trigger_ias.py` and the root `conftest.py` set `sys.dont_write_bytecode`. Elsewhere the venv's untracked `.venv/Lib/site-packages/sitecustomize.py` sets it at every start (needed for a bare `pytest`). Recreate it in a new environment with `Set-Content .venv\Lib\site-packages\sitecustomize.py "import sys; sys.dont_write_bytecode = True"`; outside the venv use `python -B` or `PYTHONDONTWRITEBYTECODE=1`.

## Running parts individually

Start only what the task needs (`node execute.mjs help`).

| Need | Bring up | Reach it at |
|---|---|---|
| A model to talk to (`illustrated_agents`, `llm-client ask`, `tiny-agent`) | `up llm-client`, then `llm-client load qwen3.5-4b` (about 35 s from cache; first time downloads) | `http://localhost:3200/v1` |
| Chat UI, backend, log search | `up`, then `load <model>` | <http://localhost:3200>, logs <http://localhost:5080> |
| Sphinx docs | `up docs` | <http://localhost:3300> |
| Tiny agent in Docker | `tiny-agent ["q"]` (needs something on 3200) | terminal |

- **The embedding container (`llama-embed`, port 3201) runs only when the task needs embeddings** (`RAGMemory`, `MemoryBank`, the embedding scripts and tests). `up llm-client` starts it too: stop it right after with `docker compose -f deploy/llm-client/docker-compose.yaml stop llama-embed`, start it on demand with `... up -d llama-embed`, and stop it again when done.
- **`llm-client` and `chat-app` share port 3200 and the GPU: run one, never both.** `down` the other first; `up all` leaves `llm-client` out.
- `docker ps` shows what is up; `down <target>` stops one part (`down all` all four); one part's `down` never touches another's containers.
- After starting a model server confirm it: `llm-client` (`* name loaded`) and `llm-client ask "…"`.
- **`WSL environment detected but no adapters were found`**: Docker lost GPU passthrough (`nvidia-smi` still works on Windows). Stop containers you care about, `wsl --shutdown`, `docker desktop start`, verify with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L`. Slow fallback: `docker compose --profile cpu up -d llama-cpu` in `deploy/llm-client`.
- `llm-client` reuses the chat app's model cache via the git-ignored `deploy/llm-client/.env` (`MODELS_CACHE=../../experimentation/chat-app/models/cache`). On a fresh clone create it from `.env.example`, or the first load downloads again.

## Tests

`node execute.mjs test` is the only entry point; never hand-roll runners. `backend`, `agent`, `llm`: pytest against fake servers. `mocked`: Playwright, API stubbed. `live`: Playwright via the gateway, real models (stack up). `all` (default) = backend + mocked (+ live if stack up); `e2e` = mocked + live.

Behaviour change = test change in the same commit. The fake router (`backend/tests/fake_router.py`) and browser stubs (`frontend/e2e/helpers.js`) mimic llama.cpp; if reality differs, fix the fake. Live tests rebuild nothing: `rebuild` after frontend/backend changes.

## Conventions

- **Windows scripting:** beyond a one-liner, write a Node `.mjs` script outside the repo and run `node file.mjs`; heredocs and here-strings break.
- **LF line endings** (`.gitattributes`); never commit CRLF `*.sh`.
- **Match surrounding code:** short *why* comments for a reader new to agents. Python stdlib-light, typed where it helps; frontend plain JS, function components, Tailwind (tokens in `frontend/src/index.css`).
- **Markdown is never hard-wrapped** (one paragraph or list item per line); the pre-commit hook unwraps staged `.md`; `node tools/unwrap-md.mjs --check|--write`; enable once: `git config core.hooksPath .githooks`.
- **Docs move with code:** a changed command, port, path or endpoint updates `INSTRUCTIONS.md` and the relevant README in the same change.
- **Do not commit** (git-ignored; check `git status`): `.env`, model files, `node_modules`, `dist`, `.venv`, `docs/_build`, `.educational-audit/`.

## Pasted model code (`experimentation/illustrated_agents/`)

Only when the user explicitly asks for a script here (see the rule at the top); otherwise a pasted snippet goes into `tiny_agent/` alone.

- **One file per concept**, grouped by folder; each pasted snippet is a **new** descriptive snake_case file. Variants of one technique share a folder (`best_of_n/orm.py`, `best_of_n/prm.py`); a comparison is its own `compare.py` there, never mixed into a concept file. A lone concept stays top level. Never edit existing scripts' code; documentation-only edits via `edu-audit` are fine.
- **A snippet about `TinyAgent` is also a request to change `tiny_agent/`.** The book's version is only the script here. Check whether `tiny_agent/` already does what the snippet shows; if not, say what is missing and change it per `tiny_agent/AGENTS.md`, with tests and docs. A script here only when explicitly asked.
- **No `__init__.py`** (namespace folder, so it never shadows the installed `illustrated_agents`).
- Each file is **self-sufficient**: own imports, `LLM(...)`, key reading, prompts, printing helpers. No shared module, no imports between scripts; duplicated setup is intended, also in `compare.py`. Copy the header of `shot_prompting/zero_shot.py`.
- Runs on the **Docker stack**, never Ollama or a new server: `base_url="http://localhost:3200/v1"`, stack model names (`qwen3.5-4b`, `gemma4-e4b`), key read at runtime from `experimentation/chat-app/secrets/llama_api_key` via `Path(__file__).parents[1]` (`[2]` in a subfolder; never print or hardcode it). Model not in `models.ini`: say so and ask.
- **Docs follow the `edu-audit` skill** (`references/style-guide.md`). Repo specifics: the module docstring names the key names (`query`, `llm` settings, helpers), the expected answer and a "Compare <script>" pointer; Google-style docstrings that pass `sphinx-build -W` (napoleon); no `Run:`/`Requires:`/`Messages sent:` sections; `Warning:` only for a real hazard (e.g. `exec` of model code); logging to stderr, never files.
- Before running: `status`; `up` if down, `load <model>` if not loaded. Must run as a file and via `trigger_ias.py <name>` (`folder/script` in a subfolder).

## Secrets and logs

- Credentials are Docker secrets: git-ignored files in `chat-app/secrets/`, generated by `up`, mounted at `/run/secrets/<name>` (OpenObserve login, `llama_api_key`). Never put a secret in compose, env vars, `.env`, an image or a log line. Services read files (`LLAMA_ARG_API_KEY_FILE`, `LLAMA_API_KEY_FILE`, Vector `SECRET[docker.<name>]`); OpenObserve cannot, so `deploy/chat-app/observability/openobserve.Dockerfile` wraps its entrypoint. The docs image writes a placeholder key file for the build and deletes it.
- OpenObserve (<http://localhost:5080>) opens logged in via the `observe` nginx proxy; anyone reaching `127.0.0.1:5080` is admin. Keep redirects relative (`absolute_redirect off`).
- All logs go to OpenObserve (containers via Vector). Only `tiny_agent/logging/log.py` writes log files (`chat-app/logs/cli.log`, `main.log`).
- `prune` removes the stack's containers, networks, `chat-app` volumes, secrets, logs, build output; keeps models and images. Never `docker system prune` (other projects live here).

## Gotchas

- **Router:** `POST /models/load` returns `{"success":true}` at once, then `loading` → `loaded` in `GET /v1/models` (`status.value`). `LLAMA_MODELS_MAX=1`: loading one unloads the other. Unknown model: 404. Web UI off.
- **Model files:** first load downloads from Hugging Face (`hf-repo` in `models.ini`) into git-ignored `models/cache/` (6 to 20 min; `prune` keeps it). Timeouts: backend `LLAMA_LOAD_TIMEOUT` 1800 s, gateway 2000 s; keep in step.
- **GPU:** 6 GB VRAM; `gemma4-e4b` uses 5.3 GB, so one model at a time, each switch a full reload (35 to 69 s). Lower `n-gpu-layers` in `models.ini` if a model does not fit.
- **Ports:** 3000, 8000, 8080, 3001 belong to other projects; gateway is **3200** (`GATEWAY_PORT`). Check `netstat` before picking one.
- **Compose:** four projects, never merged: `chat-app` (name pinned), `tiny-agent-llm-client`, `tiny-agent`, `tiny-agent-docs`; separate names keep `up`/`down --remove-orphans` from removing another's containers. The llm client and tiny agent reach the model through the published gateway (`host.docker.internal:3200`, or `TINYAGENT_LLM_URL`), so no shared network. Each `deploy/` folder stays self-sufficient: no file refers to another one. `~` is not expanded (use `USERPROFILE`/`HOME`).
- **Node on Windows:** `process.exit()` right after a timed `fetch` can crash (libuv); set `process.exitCode`. `fetch` times out after 5 min waiting for headers; long calls use `node:http`.
- **Past breakages:** OpenObserve root password needs lower, upper, digit, special (`execute.mjs` ensures it); the Hugging Face cache has symlinks Windows cannot read (size counters skip them).
- **Browser automation** (Claude in Chrome) needs the extension; otherwise API checks and Playwright.

## Before you finish

1. `node execute.mjs test`.
2. Frontend touched: `npm run build` in `frontend/`.
3. Docker or nginx touched: `docker compose config -q` in each touched `deploy/` folder, then `node execute.mjs up <target>` and, for the chat app, `health`.
4. Python touched: `uvx ruff check --select I tiny_agent experimentation` and `uvx basedpyright tiny_agent` (errors only; report any you added, not the backlog).
5. Update docs; check `git status`.
