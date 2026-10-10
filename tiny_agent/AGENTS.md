# Rules for the `tiny_agent/` folder

These apply to everything in this folder (everything under `shared/`, `main.py`, `tests/`, and anything added here). They override the repo's general guidance.

1. **Fix, do not rewrite.** When asked to fix, finish or "make it work", change only the methods and lines that need it. Keep the existing class structure, names, signatures, attributes, docstrings and style. Never replace a file wholesale, rename things, or restructure to suit your own taste. If you think the structure itself is wrong, say so and wait for a yes before changing it.
2. **Flag what is missing, do not invent it.** If something the code needs is not there (a field, method, module, import, config, dependency, test, or a decision only the owner can make), say so in your reply: what is missing, where, and why it matters. Add the smallest stub only if asked. Do not quietly fill gaps with your own design.
3. **Read the file immediately before you edit it, every time.** The owner edits these files while you work, and a file that was empty a minute ago may not be empty now. Use `Edit` on existing files, never `Write` over them. If you have not read the file's current contents, you may not overwrite it. If a write is refused because the file changed, read it again and work from what is there.
4. **Stay inside the request.** Do not touch, commit or "tidy" files you were not asked about, including empty placeholder files created for later. Mention them instead.
5. **Say what you changed, and what you did not.** End with the methods you touched, anything you left alone on purpose, and anything you could not verify.

## Current files

| File | State |
|---|---|
| `shared/tiny_agent.py` | `TinyAgent` skeleton: `run`, `_step`, `_execute_action` are placeholders. |
| `shared/response.py` | The owner's `Response` dataclass (`content`, `reasoning`, `tool_call`, `metadata`). Do not restructure it. |
| `shared/llm.py` | The owner's LLM client. Fix its methods only. |
| `shared/check.py` | Which models the server has loaded; used by `llm.py`. |
| `main.py` | Demo entry point: `python -m tiny_agent.main` from the repo root. |
| `cli.py`, `__main__.py` | The terminal chat. `tiny-agent` after `python -m pip install -e .` (see `pyproject.toml` at the repo root), or `python -m tiny_agent`. |
| `tests/` | pytest suite with a fake model server: `node execute.mjs test agent`, or `python -m pytest tiny_agent/tests` from the repo root. Add a test when you change behaviour. |
