# Installation

TinyAgent currently has no runtime dependencies beyond the Python standard library. The project was set up with Python 3.12.

## Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run the demo

```powershell
python code/main.py
```

The demo sends `Hello, agent!` to the model once and prints its reply, so it needs the Docker stack running with a model loaded (`node execute.mjs up`). It reads `TINYAGENT_MODEL` (default `gemma4-e4b`), `TINYAGENT_LLM_URL` and `TINYAGENT_API_KEY_FILE`, the same settings as the `tiny-agent` command.

## Install the terminal chat

```powershell
python -m pip install -e .
tiny-agent --help
```

This adds the ``tiny-agent`` command, still with no dependencies beyond the standard library. In this repository's virtual environment ``python -m pip`` is more reliable than the ``pip`` launcher. To chat with a model you also need the Docker stack running; see {doc}`../guides/chat-app` and ``INSTRUCTIONS.md`` in the repository root.
