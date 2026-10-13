# Rules for `tiny_agent/`

Apply to everything in this folder (`agent/`, `memory/`, `llm_client/`, `tests/`, anything added); they override the repo's general guidance.

1. **Fix, do not rewrite.** When asked to fix, finish or "make it work", change only the methods and lines that need it. Keep class structure, names, signatures, attributes, docstrings and style. Never replace a file wholesale, rename, or restructure to taste; if the structure itself seems wrong, say so and wait for a yes.
2. **Flag what is missing, do not invent it.** If something needed is absent (field, method, module, import, config, dependency, test, or an owner-only decision), say what, where and why it matters. Add the smallest stub only if asked.
3. **Read the file immediately before every edit.** The owner edits while you work. Use `Edit` on existing files, never `Write` over them; if an edit is refused because the file changed, re-read and work from what is there.
4. **Stay inside the request.** Do not touch, commit or "tidy" other files, including empty placeholders; mention them instead.
5. **Report:** the methods you touched, what you left alone on purpose, and what you could not verify.
6. **Type-checker and ruff errors are in scope** when asked to "fix errors" (`uvx basedpyright tiny_agent`, `uvx ruff check --select I tiny_agent`). Limit the fix to annotations (`dict[str, Any]` for a bare `dict`, a local variable annotation), imports and `# pyright: ignore[rule]` comments. No change to behavior, to a signature's parameters or defaults, or to what a function returns. If the only fix would change behavior or a public type (for example making `Response.content` optional), flag it under rule 2 and wait for a yes.

## Current files

| File | State |
|---|---|
| `agent/tiny_agent.py` | `TinyAgent(llm, memory)`: `run` and `_step` make one model call per task with the whole memory; `_execute_action` is still a placeholder (no tools, planner or loop yet). |
| `agent/memory.py` | `Memory`, the conversation history sent to the model (`from tiny_agent.agent.memory import Memory`; `tiny_agent.memory` re-exports it). |
| `memory/` | A package (next to `agent/`, not inside it) of `Memory` variants: `trimming.py` is `TrimmingMemory`, which keeps only the last two turns (`main.py` uses it when `TINYAGENT_TRIM_MEMORY` is on); `summarization.py` is `SummarizationMemory(llm)`, which replaces the history with a model-written running summary (`TINYAGENT_SUMMARY_MEMORY`, wins over the trim flag); `rag.py` is `RAGMemory(embedding_model, documents)`, which puts the three documents most similar to each user message in front of it (`TINYAGENT_RAG_MEMORY`, wins over both; needs the chat and the embedding model loaded together, `LLAMA_MODELS_MAX=2`). |
| `agent/trajectory.py`, `agent/step.py` | The debugging record of each run, one `Step` at a time. |
| `log.py` | The `cli.log` and `main.log` files (the only place that writes log files). |
| `llm_client/` | LLM client (`llm.py`, `response.py`, `check.py`, `env.py` and `embedding.py`, the `EmbeddingModel` that `RAGMemory` uses), with its tests in `tests/` (`test_llm.py`, `test_check.py`, `test_env.py`, `test_embedding.py`; `node execute.mjs test llm`) and its own README; same rules. See "The `llm_client/` subpackage" below. |
| `main.py` | Demo: `python -m tiny_agent.main` runs two tasks on one agent to show its memory. |
| `cli.py`, `__main__.py` | Terminal chat: `tiny-agent` after `python -m pip install -e .`, or `python -m tiny_agent.cli` (from the repo root), or `python -m tiny_agent`. |
| `tests/` | pytest with a fake model server, for the whole package: `node execute.mjs test agent` runs all but the LLM client's four files, `test llm` runs those. Includes `test_with_memory.py` and `test_without_memory.py`. Add a test when behaviour changes. |
| `.env.example`, `.env` | The settings template and the git-ignored settings file that `llm_client/env.py` reads by default (`TINYAGENT_ENV_FILE` overrides it). |

## The `llm_client/` subpackage

The rules above apply to it too, because `llm.py` and `response.py` are the owner's code. It is self-contained: it must not import from the rest of `tiny_agent` (`agent/`, `memory/`, `main.py`, `cli.py`) or the repository, only from itself (`tiny_agent.llm_client...`), and it uses only the standard library. Its tests are in `tiny_agent/tests/` with the others (`test_llm.py`, `test_check.py`, `test_env.py`, `test_embedding.py`; `node execute.mjs test llm` from the repo root). The Docker image of `deploy/llm-client` copies only this folder.
