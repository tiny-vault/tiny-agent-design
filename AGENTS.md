# AGENTS.md

Guidance for AI agents in this repo. `code/tiny_agent/` and `code/llm_client/` have stricter rules in their own `AGENTS.md` (fix methods only, never restructure, flag what is missing, re-read before every edit); read them before touching those folders.

## What this is

1. **TinyAgent** (`code/tiny_agent/`, `docs/`): minimal educational agent harness. `TinyAgent` calls the model once per task; loop, memory, tools and planning are placeholders. Beside it: the LLM client (`code/llm_client/`: `llm.py`, `response.py`, `check.py`), the `tiny-agent` terminal chat (`code/cli.py`), demo `python code/main.py`.
2. **Chat app** (`experimentation/chat-app/`): local chat UI for llama.cpp models, all in Docker; does not use `code/tiny_agent/`.

Key paths: `execute.mjs` (Docker + tests, start here), `INSTRUCTIONS.md` (how to run everything), `experimentation/docker-compose.yaml` (the stack; optional `.env` beside it), `chat-app/{frontend (React 19, Vite, Tailwind v4), backend (FastAPI), models (models.ini, cache/), deploy (Dockerfiles, nginx), observability, logs/, secrets/}`.

## Architecture

```
browser -> gateway (nginx, ONLY published port, 3200)
             |-- /          -> frontend (static)
             `-- /api, /v1  -> backend (FastAPI) -> llama.cpp (router mode, internal)
```

- Never add `ports:` to other services; add routes in `chat-app/deploy/gateway.nginx.conf`.
- Only the backend talks to llama.cpp: `/api/models`, `/api/models/load`, `/api/chat`, streamed `/v1/*` passthrough; `POST /v1/chat/completions` returns **409** unless the model is loaded.
- The frontend allows chat only when the selected model is `loaded` and no switch is in progress.
- Backend errors are always `{"error": {"message", "code"}}`; the frontend reads `message`.

## Commands (repo root)

| Task | Command |
|---|---|
| Start / stop | `node execute.mjs up` / `down` |
| Status, health, logs | `node execute.mjs status` / `health` / `logs [service]` |
| Switch model / rebuild | `node execute.mjs load <model>` / `rebuild` |
| Notebook-cell scripts | `.venv/Scripts/python trigger_ias.py [name]` (no name lists them; needs the stack up, model loaded) |
| Tests | `node execute.mjs test [backend\|agent\|llm\|mocked\|live\|e2e\|all\|list]`, runner args after `--` (e.g. `test backend -- -k load`) |
| Frontend | `cd experimentation/chat-app/frontend; npm run dev` (stack up) / `npm run build` |
| Docs | `sphinx-build -b html -n -W --keep-going docs docs/_build/html` |

Python: `.venv/Scripts/python` (3.12). Backend deps: `backend/requirements*.txt`.

## Tests

`node execute.mjs test` is the only entry point; never hand-roll runners. `backend`, `agent`, `llm`: pytest against fake servers. `mocked`: Playwright, API stubbed. `live`: Playwright through the gateway, real models (stack up). `all` (default) = backend + mocked (+ live if the stack is up); `e2e` = mocked + live.

Behaviour change means a test change in the same commit. The fake router (`backend/tests/fake_router.py`) and browser stubs (`frontend/e2e/helpers.js`) mimic real llama.cpp; if reality differs, fix the fake. Live tests rebuild nothing: `node execute.mjs rebuild` after frontend or backend changes.

## Conventions

- **Windows scripting:** beyond a one-liner, write a Node `.mjs` script (outside the repo) and run `node file.mjs`; heredocs and here-strings break here.
- **LF line endings** (`.gitattributes`); never commit CRLF `*.sh`.
- **Match surrounding code:** short comments explaining *why*, for a reader new to agents. Python stdlib-light, typed where it helps; frontend plain JS, function components, Tailwind (tokens in `frontend/src/index.css`).
- **Markdown is never hard-wrapped** (one paragraph or list item per line). Pre-commit hook unwraps staged `.md`; `node tools/unwrap-md.mjs --check|--write`. Enable once: `git config core.hooksPath .githooks`.
- **Docs move with code:** a changed command, port, path or endpoint updates `INSTRUCTIONS.md` and the relevant README in the same change.
- **Do not commit:** `.env`, model files, `node_modules`, `dist`, `.venv`, `docs/_build` (git-ignored; check `git status`).

## Pasted model code (`experimentation/illustrated_agents/`)

- Every pasted snippet becomes a **new** file there (descriptive snake_case name), unless it is another variant of a technique that already has a file: then it is one more function in that file (e.g. each Best-of-N verifier is a `best_of_n_<type>` function in `best_of_n.py`, picked by name from the command line). Otherwise never edit existing scripts.
- The folder has **no `__init__.py`** (a namespace folder), so it never shadows the installed `illustrated_agents` package; never add one.
- **Shared pieces live in `common/`** and scripts import them with `from common import ...`: `common/llm.py` (`llm`, `make_llm(temperature)`, the key from the Docker secret), `common/prompts.py` (**every prompt**, including judge prompts; import under the script's local name, e.g. `seating_query as query`), `common/output.py` (`show`, `print_diff`). Anything a second script would copy goes there, never into a script. Scripts never import each other. Copy the header of `one_shot.py`.
- `from common import` resolves because the script's folder is on `sys.path`: automatically when run as a file; `trigger_ias.py` and `docs/conf.py` add it. Keep it that way if either changes.
- It runs on the **existing Docker stack**, never Ollama (`localhost:11434`) or a new server: use `common.llm` (gateway `http://localhost:3200/v1`, the stack's model names `qwen3.5-4b`, `gemma4-e4b`, not Ollama tags, key read at runtime from `experimentation/chat-app/secrets/llama_api_key`, never printed or hardcoded). A model not in `chat-app/models/models.ini`: say so and ask.
- **One short docstring, at the top** (match `one_shot.py`): a summary line plus one paragraph covering the technique, the key names (``query``, ``llm`` settings, helpers), the expected answer and a "Compare <script>" pointer. No docstrings on variables or functions; no `Run:`/`Requires:`/`Messages sent:` sections (`trigger_ias.py` and `docs/api/illustrated-agents-scripts.rst` cover running). Add a `Warning:` only for a real hazard (e.g. `exec` of model code).
- Before running: `node execute.mjs status`; `up` if down, `load <model>` if not loaded. It must run both as a file and via `trigger_ias.py <name>`.

## Secrets and logs

- Credentials are Docker secrets: git-ignored files in `chat-app/secrets/`, generated by `up`, mounted at `/run/secrets/<name>` (OpenObserve login, `llama_api_key`). Never put a secret in compose, env vars, `.env`, an image or a log line. Services read files (`LLAMA_ARG_API_KEY_FILE`, `LLAMA_API_KEY_FILE`, Vector `SECRET[docker.<name>]`); OpenObserve cannot, so `observability/openobserve.Dockerfile` wraps its entrypoint.
- OpenObserve UI (<http://localhost:5080>) opens logged in via the `observe` nginx proxy; anyone reaching `127.0.0.1:5080` is admin. Keep redirects relative (`absolute_redirect off`).
- All logs go to OpenObserve (containers via Vector). Only `code/tiny_agent/log.py` writes files (`chat-app/logs/cli.log`, `main.log`); add no file logging elsewhere.
- `node execute.mjs prune` removes the stack's containers, networks, `chat-app` volumes, secrets, logs and build output; keeps models and images. Never `docker system prune` (other projects live here).

## Gotchas

- **Router:** `POST /models/load` returns `{"success":true}` at once; then `loading` → `loaded` in `GET /v1/models` (`status.value`). `LLAMA_MODELS_MAX=1`: loading one unloads the other. Unknown model: 404. Web UI off.
- **Model files:** first load downloads from Hugging Face (`hf-repo` in `models.ini`) into git-ignored `models/cache/` (6 to 20 min; `prune` keeps it). Timeouts: backend `LLAMA_LOAD_TIMEOUT` 1800 s, gateway 2000 s; keep in step.
- **GPU:** 6 GB VRAM; `gemma4-e4b` uses 5.3 GB, so one model at a time, every switch a full reload (35 to 69 s). Lower `n-gpu-layers` in `models.ini` if a model does not fit.
- **Ports:** 3000, 8000, 8080, 3001 belong to other projects; gateway is **3200** (`GATEWAY_PORT`). Check `netstat` before picking one.
- **Compose:** project name pinned to `chat-app`; `~` is not expanded (use `USERPROFILE`/`HOME`).
- **Node on Windows:** `process.exit()` right after a timed `fetch` can crash (libuv); set `process.exitCode`. `fetch` times out after 5 min waiting for headers; long calls use `node:http`.
- **Past breakages:** OpenObserve root password needs lower, upper, digit, special (`execute.mjs` guarantees it); the Hugging Face cache has symlinks Windows cannot read (size counters skip them).
- **Browser automation** (Claude in Chrome) needs the extension; otherwise use API checks and Playwright.

## Before you finish

1. `node execute.mjs test`.
2. Frontend touched: `npm run build` in `frontend/`.
3. Docker or nginx touched: `docker compose -f experimentation/docker-compose.yaml config -q`, then `node execute.mjs up` and `health`.
4. Update docs; check `git status`.
