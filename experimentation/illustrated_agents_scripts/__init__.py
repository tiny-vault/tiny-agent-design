"""Runnable notebook cells from *An Illustrated Guide to AI Agents*, one file per cell.

Each module here is a cell pasted from the book's notebooks, turned into a script that runs outside Jupyter. The rules every module follows:

* **Self-sufficient.** A module has its own imports, its own ``LLM(...)`` client, its own ``query`` and its own printing. No module imports another one, and there is no shared helper module, so a file can be read, copied or run on its own.
* **Docker stack, not Ollama.** Model calls go to the chat-app gateway at ``http://localhost:3200/v1`` (llama.cpp behind nginx) with the stack's model names, such as ``qwen3.5-4b``. The API key is read at runtime from the git-ignored Docker secret ``experimentation/chat-app/secrets/llama_api_key`` and is never printed.
* **Guarded entry point.** Importing a module only builds objects; the model call and the printing happen under ``if __name__ == "__main__":``. That keeps imports (for example by Sphinx autodoc) free of network calls.
* **Plain-text output.** Viewers such as ``DiffViewer`` and ``TrajectoryViewer`` render HTML only inside Jupyter, so the scripts print the plain-text equivalent.

Running a module:

* From the repo root through the trigger: ``.venv/Scripts/python trigger_ias.py <name>`` (no name lists the modules).
* From the repo root as a module: ``.venv/Scripts/python -m experimentation.illustrated_agents_scripts.<name>``.
* Directly as a file: ``.venv/Scripts/python experimentation/illustrated_agents_scripts/<name>.py``.

Modules that call a model need the stack up with the model loaded: ``node execute.mjs up`` then ``node execute.mjs load qwen3.5-4b``. The ``diff_*`` modules need no model.
"""
