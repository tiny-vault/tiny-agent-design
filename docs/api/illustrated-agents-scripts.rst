Illustrated agents scripts
==========================

Runnable cells from *An Illustrated Guide to AI Agents*, one self-sufficient
script per cell (variants of one technique share a file: ``best_of_n`` has one
function per verifier), in ``experimentation/illustrated_agents/``. Each script
has its own imports, ``LLM``, prompts and printing. Run any of
them from the repo root with ``trigger_ias.py``:

.. code-block:: console

   .venv/Scripts/python trigger_ias.py                # list the scripts
   .venv/Scripts/python trigger_ias.py shot_prompting # run one

Scripts that call a model need the Docker stack up with ``qwen3.5-4b`` loaded
(``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``); ``diff_*`` need no
model. Each script reads the key from ``chat-app/secrets/llama_api_key`` (never
printed) and calls the model only under ``if __name__ == "__main__":``, so
importing one stays offline. The folder has no ``__init__.py``, so it never
shadows the installed ``illustrated_agents`` package.

.. list-table:: Prompting scripts compared (penguin question, answer 11)
   :header-rows: 1

   * - Script
     - Technique
     - Typical prompt / completion tokens
   * - ``shot_prompting`` one-shot
     - One example with a bare answer
     - 85 / 3
   * - ``shot_prompting`` one-shot + CoT
     - One example that shows its working
     - 121 / about 110
   * - ``shot_prompting`` zero-shot + CoT
     - No example; "Let's think step by step."
     - 53 / about 150

``shot_prompting`` runs zero-, one- and few-shot, each plain and with chain-of-thought, and prints one comparison table. It replaces the book's separate one-shot, chain-of-thought one-shot and zero-shot chain-of-thought cells. The token counts above come from earlier single runs of those cells and vary between runs.

.. autosummary::
   :toctree: generated

   experimentation.illustrated_agents.shot_prompting
   experimentation.illustrated_agents.best_of_n
   experimentation.illustrated_agents.agent_without_memory
   trigger_ias
