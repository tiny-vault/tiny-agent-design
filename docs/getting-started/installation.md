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

Expected output:

```text
Received: Hello, agent!
```

This demo does not call a model.

## Install the terminal chat

```powershell
python -m pip install -e .
tiny-agent --help
```

This adds the ``tiny-agent`` command, still with no dependencies beyond the standard library. In this repository's virtual environment ``python -m pip`` is more reliable than the ``pip`` launcher. To chat with a model you also need the Docker stack running; see {doc}`../guides/chat-app` and ``INSTRUCTIONS.md`` in the repository root.
