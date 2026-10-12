# Rules for `code/tiny_agent/`

Apply to everything in this folder (`shared/`, `tests/`, anything added); they override the repo's general guidance.

1. **Fix, do not rewrite.** When asked to fix, finish or "make it work", change only the methods and lines that need it. Keep class structure, names, signatures, attributes, docstrings and style. Never replace a file wholesale, rename, or restructure to taste; if the structure itself seems wrong, say so and wait for a yes.
2. **Flag what is missing, do not invent it.** If something needed is absent (field, method, module, import, config, dependency, test, or an owner-only decision), say what, where and why it matters. Add the smallest stub only if asked.
3. **Read the file immediately before every edit.** The owner edits while you work. Use `Edit` on existing files, never `Write` over them; if an edit is refused because the file changed, re-read and work from what is there.
4. **Stay inside the request.** Do not touch, commit or "tidy" other files, including empty placeholders; mention them instead.
5. **Report:** the methods you touched, what you left alone on purpose, and what you could not verify.

## Current files

| File | State |
|---|---|
| `shared/tiny_agent.py` | `TinyAgent` skeleton: `run`, `_step`, `_execute_action` are placeholders. |
| `../llm_client/` | LLM client (`llm.py`, `response.py`, `check.py`), a self-contained top-level package with its own tests and README; same rules. |
| `../main.py` | Demo: `python code/main.py`. |
| `../cli.py`, `__main__.py` | Terminal chat: `tiny-agent` after `python -m pip install -e .`, or `python code/cli.py`, or `python -m tiny_agent`. |
| `tests/` | pytest with a fake model server: `node execute.mjs test agent`. Add a test when behaviour changes. |
