# AGENTS.md

Guidance for AI agents. `tiny_agent/` and `tiny_agent/llm_client/` have stricter `AGENTS.md` files (fix methods only, never restructure, flag what is missing, re-read before every edit); read them before touching those folders.

## What this is

1. **TinyAgent** (`tiny_agent/`, `docs/`): minimal educational agent harness; one model call per task with the conversation kept in `Memory`; the multi-step loop, tools and planning are placeholders. Also: LLM client (`tiny_agent/llm_client/`: `llm.py`, `response.py`, `check.py`), `tiny-agent` terminal chat (`cli.py`), demo `python -m tiny_agent.main`.
2. **Chat app** (`experimentation/chat-app/`): local chat UI for llama.cpp models, all in Docker; does not use `tiny_agent/`.

Key paths: `execute.mjs` (Docker + tests, start here), `INSTRUCTIONS.md` (how to run everything), `deploy/` (everything Docker, four independent parts, each folder with its own `docker-compose.yaml`, Dockerfiles, `.env.example` and README: `chat-app/` (llama.cpp, backend, frontend, gateway, `observability/`, `init-secrets.mjs`), `llm-client/`, `tiny-agent/` (both run on demand), `docs/` (Sphinx)), `experimentation/chat-app/{frontend (React 19, Vite, Tailwind v4), backend (FastAPI), models (models.ini, cache/), logs/, secrets/}` (source and state; the Docker files are in `deploy/chat-app/`).

## Architecture

```
browser -> gateway (nginx, ONLY published port, 3200)
             |-- /          -> frontend (static)
             `-- /api, /v1  -> backend (FastAPI) -> llama.cpp (router mode, internal)
```

- No `ports:` on other services; add routes in `deploy/chat-app/gateway.nginx.conf`.
- Only the backend talks to llama.cpp: `/api/models`, `/api/models/load`, `/api/chat`, streamed `/v1/*` passthrough; `POST /v1/chat/completions` is **409** unless the model is loaded.
- Frontend allows chat only when the selected model is `loaded` and no switch is in progress.
- Backend errors are always `{"error": {"message", "code"}}`; the frontend reads `message`.

## Commands (repo root)

| Task | Command |
|---|---|
| Start / stop | `node execute.mjs up [chat-app\|llm-client\|tiny-agent\|docs\|all]` / `down [target]` (default `chat-app`; each target is its own compose project in `deploy/<target>/`) |
| LLM client / tiny agent in Docker | `up llm-client` starts the client folder's **own model server** on `localhost:3200` (same port and GPU as the chat app: run one or the other; the `illustrated_agents` scripts run against it as they are). Then `node execute.mjs llm-client [model]` (which models are loaded) / `llm-client load <model>` / `llm-client ask "question"` (really talks to the model), and `tiny-agent ["question"]`, which uses whatever answers on port 3200 (that server or the chat app's gateway), or `TINYAGENT_LLM_URL` |
| Status, health, logs | `node execute.mjs status` / `health` / `logs [service]` |
| Switch model / rebuild | `node execute.mjs load <model>` / `rebuild` |
| Notebook-cell scripts | `.venv/Scripts/python experimentation/trigger_ias.py [name]` (no name opens a popup REPL to pick and run; `--list` prints them; stack up, model loaded) |
| Tests | `node execute.mjs test [backend\|agent\|llm\|mocked\|live\|e2e\|ias\|all\|list]`, runner args after `--` (e.g. `test backend -- -k load`) |
| Frontend | `cd experimentation/chat-app/frontend; npm run dev` (stack up) / `npm run build` |
| Docs | `node execute.mjs docs` builds them into an image and serves them on <http://localhost:3300> (`DOCS_PORT`); without Docker: `docs --local [--no-build] [--port N]`; build only: `sphinx-build -b html -n -W --keep-going docs docs/_build/html` |
| Educational audit (Claude skill) | `/edu-audit <folder>` (`.claude/skills/edu-audit/`, git-ignored, local) |

Python: `.venv/Scripts/python` (3.12). Backend deps: `backend/requirements*.txt`. **No `__pycache__` in the repo:** `execute.mjs` and the Dockerfiles set `PYTHONDONTWRITEBYTECODE=1`, `trigger_ias.py` and the root `conftest.py` set `sys.dont_write_bytecode`. Everything else relies on the virtual environment: `.venv/Lib/site-packages/sitecustomize.py` (untracked, because `.venv` is) sets `sys.dont_write_bytecode = True` at every Python start, which is the only thing that also covers a bare `pytest` (it compiles the root `conftest.py` before that file can set the flag). Recreate it in a new environment with `Set-Content .venv\Lib\site-packages\sitecustomize.py "import sys; sys.dont_write_bytecode = True"`; outside the venv use `python -B` or set `PYTHONDONTWRITEBYTECODE=1`.

## Bringing parts up individually

Four parts, each a folder in `deploy/` with its own compose file, README and `.env.example`; start only what the task needs (`node execute.mjs help` lists everything). Each can also be run by hand from its folder with `docker compose`.

| Need | Bring up | Reach it at |
|---|---|---|
| Just a model to talk to (run the `illustrated_agents` scripts, `llm-client ask`, `tiny-agent`) | `node execute.mjs up llm-client`, then `llm-client load qwen3.5-4b` (about 35 s from the cache; first time it downloads) | `http://localhost:3200/v1` |
| The chat UI, backend and log search | `node execute.mjs up` (= `up chat-app`), then `load <model>` | <http://localhost:3200>, logs <http://localhost:5080> |
| The Sphinx docs | `node execute.mjs up docs` | <http://localhost:3300> |
| The tiny agent in Docker | `node execute.mjs tiny-agent ["question"]` (needs something answering on port 3200) | the terminal |

- **`llm-client` and `chat-app` use the same port (3200) and the same GPU: run one, never both.** Stop the other first with `node execute.mjs down <target>`; `up all` leaves `llm-client` out for this reason.
- Check what is up with `docker ps`; stop a part with `down <target>` (`down all` stops all four). Nothing in one part's `down` touches another part's containers.
- After starting a model server, confirm the model: `node execute.mjs llm-client` (status list, `* name loaded`) and `llm-client ask "…"` (a real answer).
- **If a model container will not start with `WSL environment detected but no adapters were found`**: Docker lost GPU passthrough (Windows still sees the GPU in `nvidia-smi`). Make sure no container you care about is running, then `wsl --shutdown`, `docker desktop start`, and check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L` before retrying. `docker compose --profile cpu up -d llama-cpu` in `deploy/llm-client` is a slow CPU fallback.
- `llm-client` reuses the chat app's downloaded models through `deploy/llm-client/.env` (`MODELS_CACHE=../../experimentation/chat-app/models/cache`, git-ignored). On a fresh clone create it from `.env.example`, or the first load downloads the model again.

## Tests

`node execute.mjs test` is the only entry point; never hand-roll runners. `backend`, `agent`, `llm`: pytest against fake servers. `mocked`: Playwright, API stubbed. `live`: Playwright via the gateway, real models (stack up). `all` (default) = backend + mocked (+ live if stack up); `e2e` = mocked + live.

Behaviour change = test change in the same commit. The fake router (`backend/tests/fake_router.py`) and browser stubs (`frontend/e2e/helpers.js`) mimic real llama.cpp; if reality differs, fix the fake. Live tests rebuild nothing: `node execute.mjs rebuild` after frontend/backend changes.

## Conventions

- **Windows scripting:** beyond a one-liner, write a Node `.mjs` script (outside the repo) and run `node file.mjs`; heredocs and here-strings break.
- **LF line endings** (`.gitattributes`); never commit CRLF `*.sh`.
- **Match surrounding code:** short *why* comments for a reader new to agents. Python stdlib-light, typed where it helps; frontend plain JS, function components, Tailwind (tokens in `frontend/src/index.css`).
- **Markdown is never hard-wrapped** (one paragraph or list item per line); pre-commit hook unwraps staged `.md`; `node tools/unwrap-md.mjs --check|--write`; enable once: `git config core.hooksPath .githooks`.
- **Docs move with code:** a changed command, port, path or endpoint updates `INSTRUCTIONS.md` and the relevant README in the same change.
- **Do not commit** (git-ignored; check `git status`): `.env`, model files, `node_modules`, `dist`, `.venv`, `docs/_build`, `.educational-audit/`.

## Pasted model code (`experimentation/illustrated_agents/`)

- **One file per concept**, grouped by folder: each pasted snippet is a **new** file (descriptive snake_case). Variants of one technique live in the same folder (`best_of_n/orm.py`, `best_of_n/prm.py`, `shot_prompting/zero_shot.py`), one file each. If a folder has a comparison, it is its own `compare.py` in that folder, never mixed into a concept file. A lone concept stays at the top level (`agent_without_memory.py`). Never edit existing scripts' code; documentation-only edits via `edu-audit` are fine.
- **A snippet about `TinyAgent` is also a request to change `code/`.** If a pasted snippet builds or extends the agent (e.g. `TinyAgent(llm=llm, memory=memory)`), the book's version is only the script here; the repo's own `TinyAgent` lives in `tiny_agent/`. Check whether `tiny_agent/` already does what the snippet shows. If not, say what is missing and make the change there, following `tiny_agent/AGENTS.md` (fix methods only, never restructure, flag what is missing), with its tests and docs in the same change. The script in `illustrated_agents/` is still created as usual.
- **No `__init__.py`** (namespace folder, so it never shadows the installed `illustrated_agents` package).
- Each file is **self-sufficient**: own imports, `LLM(...)`, key reading, prompts (`query`, judge prompts), printing helpers (`show`). No shared module (no `common/`), no imports between scripts; duplicated setup is intended, including in `compare.py`. Copy the header of `shot_prompting/zero_shot.py`.
- Runs on the **existing Docker stack**, never Ollama (`localhost:11434`) or a new server: `base_url="http://localhost:3200/v1"`, stack model names (`qwen3.5-4b`, `gemma4-e4b`), key read at runtime from `experimentation/chat-app/secrets/llama_api_key` via `Path(__file__).parents[1]` (`parents[2]` in a subfolder; never print or hardcode it). Model not in `chat-app/models/models.ini`: say so and ask.
- **Documentation follows the `edu-audit` skill** (rules: its `references/style-guide.md`), for new snippets too. Repo specifics: the module docstring names the key names (``query``, ``llm`` settings, helpers), the expected answer and a "Compare <script>" pointer; Google-style docstrings that pass `sphinx-build -W` (napoleon); no `Run:`/`Requires:`/`Messages sent:` sections (`trigger_ias.py` and `docs/api/illustrated-agents-scripts.rst` cover running); `Warning:` only for a real hazard (e.g. `exec` of model code); logging goes to stderr, never to files.
- Before running: `node execute.mjs status`; `up` if down, `load <model>` if not loaded. Must run as a file and via `trigger_ias.py <name>` (`folder/script` for a script in a subfolder, e.g. `best_of_n/orm`).

## Secrets and logs

- Credentials are Docker secrets: git-ignored files in `chat-app/secrets/`, generated by `up`, mounted at `/run/secrets/<name>` (OpenObserve login, `llama_api_key`). Never put a secret in compose, env vars, `.env`, an image or a log line. Services read files (`LLAMA_ARG_API_KEY_FILE`, `LLAMA_API_KEY_FILE`, Vector `SECRET[docker.<name>]`); OpenObserve cannot, so `deploy/chat-app/observability/openobserve.Dockerfile` wraps its entrypoint. The docs image needs no secret: it writes a placeholder key file for the build (the scripts only read it on import) and deletes it.
- OpenObserve UI (<http://localhost:5080>) opens logged in via the `observe` nginx proxy; anyone reaching `127.0.0.1:5080` is admin. Keep redirects relative (`absolute_redirect off`).
- All logs go to OpenObserve (containers via Vector). Only `tiny_agent/log.py` writes log files (`chat-app/logs/cli.log`, `main.log`).
- `node execute.mjs prune` removes the stack's containers, networks, `chat-app` volumes, secrets, logs, build output; keeps models and images. Never `docker system prune` (other projects live here).

## Gotchas

- **Router:** `POST /models/load` returns `{"success":true}` at once, then `loading` → `loaded` in `GET /v1/models` (`status.value`). `LLAMA_MODELS_MAX=1`: loading one unloads the other. Unknown model: 404. Web UI off.
- **Model files:** first load downloads from Hugging Face (`hf-repo` in `models.ini`) into git-ignored `models/cache/` (6 to 20 min; `prune` keeps it). Timeouts: backend `LLAMA_LOAD_TIMEOUT` 1800 s, gateway 2000 s; keep in step.
- **GPU:** 6 GB VRAM; `gemma4-e4b` uses 5.3 GB, so one model at a time, each switch a full reload (35 to 69 s). Lower `n-gpu-layers` in `models.ini` if a model does not fit.
- **Ports:** 3000, 8000, 8080, 3001 belong to other projects; gateway is **3200** (`GATEWAY_PORT`). Check `netstat` before picking one.
- **Compose:** four projects, never merged into one file: `chat-app` (name pinned; the folder is `deploy/chat-app`), `tiny-agent-llm-client`, `tiny-agent`, `tiny-agent-docs`. Separate names keep `up`/`down --remove-orphans` of one from removing another's containers. The llm client and the tiny agent reach the model through the published gateway (`host.docker.internal:3200`, or `TINYAGENT_LLM_URL`), so they need no shared network. Each folder must stay self-sufficient: no file in it refers to another `deploy/` folder. `~` is not expanded (use `USERPROFILE`/`HOME`).
- **Node on Windows:** `process.exit()` right after a timed `fetch` can crash (libuv); set `process.exitCode`. `fetch` times out after 5 min waiting for headers; long calls use `node:http`.
- **Past breakages:** OpenObserve root password needs lower, upper, digit, special (`execute.mjs` ensures it); the Hugging Face cache has symlinks Windows cannot read (size counters skip them).
- **Browser automation** (Claude in Chrome) needs the extension; otherwise API checks and Playwright.

## Before you finish

1. `node execute.mjs test`.
2. Frontend touched: `npm run build` in `frontend/`.
3. Docker or nginx touched: `docker compose config -q` inside each touched folder of `deploy/` (`chat-app`, `llm-client`, `tiny-agent`, `docs`), then `node execute.mjs up <target>` and, for the chat app, `health`.
4. Python touched: `uvx ruff check --select I code experimentation` and `uvx basedpyright code` (errors only; the repo has a backlog, so report any error you added, not the whole list).
5. Update docs; check `git status`.
