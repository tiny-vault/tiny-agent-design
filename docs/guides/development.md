# Development

## Build the documentation

```powershell
pip install -r docs/requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser, or run `node execute.mjs docs` from the repository root to build the docs into a Docker image and serve them at <http://localhost:3300> (`DOCS_PORT` changes the port; `node execute.mjs down docs` stops them). Without Docker, `node execute.mjs docs --local` builds with the Sphinx on this machine and serves the result (Ctrl-C stops it; `--no-build` serves the last build; `--port N` changes the port). The command treats warnings as errors and checks cross-references (`-n`). In this repository's virtual environment the `sphinx-build` launcher exits silently, so `python -m sphinx` is the reliable form.

## Run the tests

One command runs everything, from the repository root:

```powershell
node execute.mjs test            # backend + tiny_agent + mocked (+ live when the stack is up)
node execute.mjs test agent      # only the tiny_agent tests
node execute.mjs test llm        # only the llm_client tests
node execute.mjs test list       # list the tests without running them
```

| Target | What it covers | Needs |
|---|---|---|
| `backend` | the FastAPI service, against a fake llama.cpp router | nothing |
| `llm` | `llm_client` (the LLM client, `Response`, the loaded-model check), against a fake model server | nothing |
| `agent` | `tiny_agent` (chat command, logs, demo, the harness), against a fake model server | nothing |
| `mocked` | the React app in a real browser with the API stubbed | nothing |
| `live` | real model switching and OpenObserve, in a real browser | the Docker stack running |

Runner arguments go after `--`, for example `node execute.mjs test backend -- -k load`. When behaviour changes, add or update a test in the same change.

## Docstring style

Use Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Example:`). They are rendered by `sphinx.ext.napoleon`. API pages are generated from the source by `autosummary`, so new modules need only be added to `docs/api/index.rst`.

## Markdown style

Markdown is never hard-wrapped: one paragraph or list item is one line. A pre-commit hook (`.githooks/pre-commit`) unwraps staged files, and `node tools/unwrap-md.mjs --check` checks the whole repository. Enable the hook once per clone with `git config core.hooksPath .githooks`.

## Known documentation gaps

* `_step` and `_execute_action` are private and are not in the API reference. Their roles are described in the architecture overview.
* `llm_client.env` (the `.env` loader) is not in the API reference.
