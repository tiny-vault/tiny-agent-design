"""Runnable notebook cells from *An Illustrated Guide to AI Agents*, one script per cell.

Each module is self-sufficient (own imports, ``LLM``, ``query``, printing), calls the Docker stack at ``http://localhost:3200/v1`` with the key from ``chat-app/secrets/llama_api_key`` (never printed), and calls the model only under ``if __name__ == "__main__":``, so imports stay offline. Jupyter-only viewers are printed as plain text.

Run from the repo root: ``.venv/Scripts/python trigger_ias.py <name>`` (no name lists them). Model scripts need ``node execute.mjs up`` and ``node execute.mjs load qwen3.5-4b``; ``diff_*`` need no model.
"""
