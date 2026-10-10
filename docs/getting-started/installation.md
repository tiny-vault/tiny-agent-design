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
python -m tiny_agent.main
```

Expected output:

```text
Received: Hello, agent!
```

This demo does not call a model. To chat with one, install the ``tiny-agent`` command; see ``INSTRUCTIONS.md`` in the repository root.
