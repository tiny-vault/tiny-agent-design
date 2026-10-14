# Rules for `tiny_agent/`

Apply to everything in this folder (`components/`, `config/`, `tests/`, anything added); they override the repo's general guidance.

1. **Fix, do not rewrite.** When asked to fix, finish or "make it work", change only the methods and lines that need it. Keep class structure, names, signatures, attributes, docstrings and style. Never replace a file wholesale, rename, or restructure to taste; if the structure itself seems wrong, say so and wait for a yes.
2. **Flag what is missing, do not invent it.** If something needed is absent (field, method, module, import, config, dependency, test, or an owner-only decision), say what, where and why it matters. Add the smallest stub only if asked.
3. **Read the file immediately before every edit.** The owner edits while you work. Use `Edit` on existing files, never `Write` over them; if an edit is refused because the file changed, re-read and work from what is there.
4. **Stay inside the request.** Do not touch, commit or "tidy" other files, including empty placeholders; mention them instead.
5. **Report:** the methods you touched, what you left alone on purpose, and what you could not verify.
6. **Type-checker and ruff errors are in scope** when asked to "fix errors" (`uvx basedpyright tiny_agent`, `uvx ruff check --select I tiny_agent`). Limit the fix to annotations (`dict[str, Any]` for a bare `dict`, a local variable annotation), imports and `# pyright: ignore[rule]` comments. No change to behavior, to a signature's parameters or defaults, or to what a function returns. If the only fix would change behavior or a public type (for example making `Response.content` optional), flag it under rule 2 and wait for a yes.

## Code layout

One package (`agent.py` at the top, the parts in `components/`), no separate "base" and "inherit" folders. A base class is used only where there are several real variants that share code: `components/memory/memory.py` `BaseMemory` and `components/tools/tools.py` `BaseTools`. Everything else is a plain class. A new variant is a subclass in the same folder (use `@override`).

| Concept | File | What it is |
|---|---|---|
| Agent | `agent.py` `TinyAgent(llm, memory, tools=None)` | One model call per task with the whole memory; with `tools` the tool prompt is a system message, a tool call in the reply is parsed and `_execute_action` runs it and returns the observation (no planner or loop yet) |
| Memory | `components/memory/memory.py` `BaseMemory`: message list, `get_messages`, `_append`; a subclass writes `add` | `memory.py` `Memory` (keeps all); `trimming.py` `TrimmingMemory` (last two turns, `TINYAGENT_TRIM_MEMORY`); `summarization.py` `SummarizationMemory(llm)` (`TINYAGENT_SUMMARY_MEMORY`); `rag.py` `RAGMemory(embedding_model, documents)` (top 3 documents, `TINYAGENT_RAG_MEMORY`); `memory_bank.py` `MemoryBank(llm, embedding_model)` (fading summaries and a user portrait, `TINYAGENT_BANK_MEMORY`); `similarity.py` `cosine`. Flag priority: bank > rag > summary > trim; rag and bank need chat and embedding models together (`LLAMA_MODELS_MAX=2`) |
| Tools | `components/tools/tools.py` `BaseTools`: registry, `add_tool`, `execute` with approval, default `observation` and `is_done`; a subclass writes `prompt` and `parse` (and `schemas` for native calling) | `prompting.py` `PromptingTools` (definitions in the system prompt, JSON-in-text); `learned.py` `LearnedTools` (no tool text in the prompt, definitions in the request's `tools` field via `schemas`, structured call parsed); `multiply.py`; `default.py` `default_tools()` |
| Trajectory and step | `components/trajectory.py` `Trajectory`: `runs`, `initialize`, `add`; `components/step.py` `Step` (thought, action, observation, answer, metadata) | The logbook for the developer, not the model |
| Model client | `components/client/`: `llm.py` `LLM`, `embedding.py` `EmbeddingModel`, `http.py` `post_json` (the one HTTP POST), `check.py` (which models are loaded; also `python -m tiny_agent.components.client.check`), own README. `response.py` `Response`: what a reply becomes | Plain classes, no base class |

## Everything else

| File | State |
|---|---|
| `config/` | `env.py`: loads the `.env` file, `api_key_from_env()`. `settings.py`: `Settings.from_env()`, every other `TINYAGENT_*` value with one default each; code asks `Settings`, never `os.environ` (a new setting is a new field there, plus a line in `.env.example`). |
| `logging/log.py` | Writes `cli.log` and `main.log`, the only log files. |
| `main.py`, `cli.py`, `__main__.py` | Demo (`python -m tiny_agent.main`) and terminal chat (`tiny-agent` after `python -m pip install -e .`, or `python -m tiny_agent.cli`, or `python -m tiny_agent`). |
| `tests/` | pytest with a fake model server. `node execute.mjs test agent` runs all but the four LLM client files (`test_llm.py`, `test_check.py`, `test_settings.py`, `test_embedding.py`), `test llm` runs those. Add a test when behaviour changes. |
| `.env.example`, `.env` | Settings template and the git-ignored file `config/env.py` reads (`TINYAGENT_ENV_FILE` overrides). |

## The model client (`components/client/`, `config/`)

Same rules, because `llm.py` and `response.py` are the owner's code. Libraries: pydantic, pydantic-settings, httpx and click (the `pyproject.toml` dependencies; the two Docker images `pip install` them). These files import only each other, never the rest of `components/` (memory, tools), `agent.py`, `main.py`, `cli.py` or the repository. The Docker image of `deploy/llm-client` copies exactly these files; a new import here means a new `COPY` line there.
