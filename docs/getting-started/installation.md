# Installation

TinyAgent currently has no runtime dependencies beyond the Python standard
library. The project was set up with Python 3.12.

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
python main.py
```

Expected output:

```text
Received: Hello, agent!
```
