# Rules for the `code/llm_client/` folder

The same rules as `code/tiny_agent/AGENTS.md` apply here, because `llm.py` and `response.py` are the owner's code:

1. **Fix, do not rewrite.** Change only the methods and lines that need it. Keep the class structure, names, signatures, attributes, docstrings and style.
2. **Flag what is missing, do not invent it.** Say what is missing, where and why; add the smallest stub only if asked.
3. **Read the file immediately before you edit it, every time.** Use `Edit` on existing files, never `Write` over them.
4. **Stay inside the request.** Do not touch files you were not asked about.
5. **Say what you changed, and what you did not.**

This folder is self-contained: it must not import anything from `tiny_agent` or the rest of the repository, and it uses only the standard library. Its tests are in `tests/` (`node execute.mjs test llm` from the repo root).
