# Rules for `code/tiny_agent/`

Apply to everything in this folder (`shared/`, `tests/`, anything added); they override the repo's general guidance.

1. **Fix, do not rewrite.** When asked to fix, finish or "make it work", change only the methods and lines that need it. Keep class structure, names, signatures, attributes, docstrings and style. Never replace a file wholesale, rename, or restructure to taste; if the structure itself seems wrong, say so and wait for a yes.
2. **Flag what is missing, do not invent it.** If something needed is absent (field, method, module, import, config, dependency, test, or an owner-only decision), say what, where and why it matters. Add the smallest stub only if asked.
3. **Read the file immediately before every edit.** The owner edits while you work. Use `Edit` on existing files, never `Write` over them; if an edit is refused because the file changed, re-read and work from what is there.
4. **Stay inside the request.** Do not touch, commit or "tidy" other files, including empty placeholders; mention them instead.
5. **Report:** the methods you touched, what you left alone on purpose, and what you could not verify.
6. **Type-checker and ruff errors are in scope** when asked to "fix errors" (`uvx basedpyright code`, `uvx ruff check --select I code`). Limit the fix to annotations (`dict[str, Any]` for a bare `dict`, a local variable annotation), imports and `# pyright: ignore[rule]` comments. No change to behavior, to a signature's parameters or defaults, or to what a function returns. If the only fix would change behavior or a public type (for example making `Response.content` optional), flag it under rule 2 and wait for a yes.

## Current files

| File | State |
|---|---|
| `shared/tiny_agent.py` | `TinyAgent` skeleton: `run`, `_step`, `_execute_action` are placeholders. |
| `../llm_client/` | LLM client (`llm.py`, `response.py`, `check.py`), a self-contained top-level package with its own tests and README; same rules. |
| `../main.py` | Demo: `python code/main.py`. |
| `../cli.py`, `__main__.py` | Terminal chat: `tiny-agent` after `python -m pip install -e .`, or `python code/cli.py`, or `python -m tiny_agent`. |
| `tests/` | pytest with a fake model server: `node execute.mjs test agent`. Add a test when behaviour changes. |
