Illustrated agents scripts
==========================

Runnable cells from *An Illustrated Guide to AI Agents*, one script per cell
(variants of one technique share a file, like ``best_of_n``), in
``experimentation/illustrated_agents/``. Shared pieces live in its ``common``
package: the stack's ``llm``, every prompt (``common.prompts``) and plain-text
printers. Run any of
them from the repo root with ``trigger_ias.py``:

.. code-block:: console

   .venv/Scripts/python trigger_ias.py                # list the scripts
   .venv/Scripts/python trigger_ias.py cot_one_shot   # run one

Scripts that call a model need the Docker stack up with ``qwen3.5-4b`` loaded
(``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``); ``diff_*`` need no
model. ``common.llm`` reads the key from ``chat-app/secrets/llama_api_key`` (never
printed); each script calls the model only under ``if __name__ == "__main__":``, so
importing one stays offline. The folder has no ``__init__.py``, so it never
shadows the installed ``illustrated_agents`` package.

.. list-table:: Prompting scripts compared (penguin question, answer 11)
   :header-rows: 1

   * - Script
     - Technique
     - Typical prompt / completion tokens
   * - ``one_shot``
     - One example with a bare answer
     - 85 / 3
   * - ``cot_one_shot``
     - One example that shows its working
     - 121 / about 110
   * - ``zero_shot_cot``
     - No example; "Let's think step by step."
     - 53 / about 150
   * - ``zero_shot_cot_raw``
     - Same as ``zero_shot_cot``, prints the raw ``Response``
     - 53 / about 150

.. autosummary::
   :toctree: generated

   experimentation.illustrated_agents.one_shot
   experimentation.illustrated_agents.cot_one_shot
   experimentation.illustrated_agents.zero_shot_cot
   experimentation.illustrated_agents.zero_shot_cot_raw
   experimentation.illustrated_agents.self_consistency
   experimentation.illustrated_agents.best_of_n
   experimentation.illustrated_agents.run_agent_trajectory
   experimentation.illustrated_agents.diff_ch1_ch2
   experimentation.illustrated_agents.diff_ch1_ch6
   common
   common.llm
   common.prompts
   common.output
   trigger_ias
