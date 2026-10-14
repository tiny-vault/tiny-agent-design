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
| Agent | `agent.py` `TinyAgent(llm, memory, tools=None)` | One model call per task with the whole memory; with `tools` the tool prompt is a system message, a tool call in the reply is parsed and `_execute_action` runs it and returns the observation (one step per run; with `planner=ReAct(...)` `run` loops until `final_answer` or `max_steps`) |
| Memory | `components/memory/memory.py` `BaseMemory`: message list, `get_messages`, `_append`; a subclass writes `add` | `memory.py` `Memory` (keeps all); `trimming.py` `TrimmingMemory` (last two turns, `TINYAGENT_TRIM_MEMORY`); `summarization.py` `SummarizationMemory(llm)` (`TINYAGENT_SUMMARY_MEMORY`); `rag.py` `RAGMemory(embedding_model, documents)` (top 3 documents, `TINYAGENT_RAG_MEMORY`); `memory_bank.py` `MemoryBank(llm, embedding_model)` (fading summaries and a user portrait, `TINYAGENT_BANK_MEMORY`); `similarity.py` `cosine`. Flag priority: bank > rag > summary > trim; rag and bank need chat and embedding models together (`LLAMA_MODELS_MAX=2`) |
| Tools | `components/tools/tools.py` `BaseTools`: registry, `add_tool`, `execute` with approval, default `observation` and `is_done`; a subclass writes `prompt` and `parse` (and `schemas` for native calling) | `prompting.py` `PromptingTools` (definitions in the system prompt, JSON-in-text); `learned.py` `LearnedTools` (no tool text in the prompt, definitions in the request's `tools` field via `schemas`, structured call parsed); `functions/` (`add.py`, `subtract.py`, `multiply.py`, `divide.py`, one tool per file); `default.py` `default_tools()` |
| Planning | `components/planning/`: `react.py` `ReAct(max_steps)`: `prompt` (THOUGHT/ACTION format), `parse` (thought to `reasoning`, action to `content`; prompting tools only); `native_react.py` `NativeReAct(ReAct)`: empty `prompt`, pass-through `parse` (native thinking and learned tools). Both re-exported from `planning/__init__.py` | `tests/test_react.py` |
| Trajectory and step | `components/trajectory.py` `Trajectory`: `runs`, `initialize`, `add`; `components/step.py` `Step` (thought, action, observation, answer, metadata) | The logbook for the developer, not the model |
| Evaluation | `components/evaluator.py` `Scorer` (`(prediction, example) -> bool \| float`), `Benchmark(name, examples, scorer)` dataclass; `exact_match_scorer` (first standalone letter A to J vs `expected`); `Evaluator(create_agent).run(benchmark)` (a fresh agent per example, returns `name`, `pass_rate`, `results`). `components/benchmarks.py` `mmlu_pro` (three questions) | `tests/test_evaluator.py`; real run `complete/evaluation.py` |
| Model client | `components/client/`: `llm.py` `LLM`, `embedding.py` `EmbeddingModel`, `http.py` `post_json` (the one HTTP POST), `check.py` (which models are loaded; also `python -m tiny_agent.components.client.check`), own README. `response.py` `Response`: what a reply becomes | Plain classes, no base class |

## Everything else

| File | State |
|---|---|
| `config/` | `env.py`: loads the `.env` file, `api_key_from_env()`. `settings.py`: `Settings.from_env()`, every other `TINYAGENT_*` value with one default each; code asks `Settings`, never `os.environ` (a new setting is a new field there, plus a line in `.env.example`). |
| `logging/log.py` | Writes `cli.log` and `main.log`, the only log files. |
| `main.py`, `cli.py`, `__main__.py` | Demo (`python -m tiny_agent.main`) and terminal chat (`tiny-agent` after `python -m pip install -e .`, or `python -m tiny_agent.cli`, or `python -m tiny_agent`). |
| `tests/` | pytest with a fake model server. One file per concept: `test_agent.py` (harness, memory across runs, step), `test_tools.py` (prompting and learned), `test_memory.py` (trimming, summarization, RAG, bank; one parametrized contract for all four), `test_client.py` (LLM, check, embedding, settings), `test_cli.py` (chat and log files), `test_complete.py` (the real-model runs in `complete/`). Shared stand-ins and helpers live in `stubs.py` (`ScriptedLLM`, `EchoLLM`, `RecordingLLM`, `NamesLLM`, `texts`, `llm`, `write_env`), the real-model fixtures (`embedder`, `chat_model`) in `conftest.py`: reuse them, do not write another stub. `node execute.mjs test agent` runs everything but `test_client.py`, `test llm` runs that. Add a test when behaviour changes. |
| `complete/` | Real-model runs, no fakes: `happy_path.py` (multiply by prompting, then divide by learned calling) , `full_run.py` (one scenario per part) and `evaluation.py` (`mmlu_pro` with a fresh `NativeReAct` agent per question, prints the report; `python -m tiny_agent.complete.evaluation`). `python -m tiny_agent.complete.happy_path`; tested by `tests/test_complete.py`, skipped when no model is loaded. |
| `.env.example`, `.env` | Settings template and the git-ignored file `config/env.py` reads (`TINYAGENT_ENV_FILE` overrides). |

## The model client (`components/client/`, `config/`)

Same rules, because `llm.py` and `response.py` are the owner's code. Libraries: pydantic, pydantic-settings, httpx and click (the `pyproject.toml` dependencies; the two Docker images `pip install` them). These files import only each other, never the rest of `components/` (memory, tools), `agent.py`, `main.py`, `cli.py` or the repository. The Docker image of `deploy/llm-client` copies exactly these files; a new import here means a new `COPY` line there.
