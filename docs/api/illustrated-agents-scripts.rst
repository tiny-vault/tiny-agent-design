Illustrated agents scripts
==========================

Runnable cells from *An Illustrated Guide to AI Agents*, one self-sufficient
script per concept, in ``experimentation/illustrated_agents/``. Related scripts
share a folder (``shot_prompting/``, ``best_of_n/``) and each folder has a
``compare.py`` that runs the concepts side by side. Each script
has its own imports, ``LLM``, prompts and printing. Run any of
them from the repo root with ``experimentation/trigger_ias.py``:

.. code-block:: console

   .venv/Scripts/python experimentation/trigger_ias.py                          # popup REPL: pick a script, it runs, the popup returns
   .venv/Scripts/python experimentation/trigger_ias.py --list                   # list the scripts
   .venv/Scripts/python experimentation/trigger_ias.py shot_prompting/zero_shot # run one and exit

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
   * - ``shot_prompting/one_shot``
     - One example with a bare answer
     - 85 / 3
   * - ``shot_prompting/one_shot`` (CoT)
     - One example that shows its working
     - 121 / about 110
   * - ``shot_prompting/zero_shot`` (CoT)
     - No example; "Let's think step by step."
     - 53 / about 150

Each of ``zero_shot``, ``one_shot`` and ``few_shot`` runs its technique plain and with chain-of-thought; ``shot_prompting/compare`` runs all six and prints one table. The token counts above come from earlier single runs of the book's cells and vary between runs.

``best_of_n/`` has one script per verifier (``unit_tests``, ``orm``, ``prm``, ``majority_vote``, ``weighted_prm``); ``best_of_n/compare`` samples candidates once and lets all five pick from the same pool.

.. autosummary::
   :toctree: generated

   experimentation.illustrated_agents.shot_prompting.zero_shot
   experimentation.illustrated_agents.shot_prompting.one_shot
   experimentation.illustrated_agents.shot_prompting.few_shot
   experimentation.illustrated_agents.shot_prompting.compare
   experimentation.illustrated_agents.best_of_n.unit_tests
   experimentation.illustrated_agents.best_of_n.orm
   experimentation.illustrated_agents.best_of_n.prm
   experimentation.illustrated_agents.best_of_n.majority_vote
   experimentation.illustrated_agents.best_of_n.weighted_prm
   experimentation.illustrated_agents.best_of_n.compare
   experimentation.illustrated_agents.agent_without_memory
   experimentation.trigger_ias
