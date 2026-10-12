Illustrated agents scripts
==========================

Runnable cells from *An Illustrated Guide to AI Agents*, one self-sufficient
script per cell, in ``experimentation/illustrated_agents_scripts/``. Run any of
them from the repo root with ``trigger_ias.py``:

.. code-block:: console

   .venv/Scripts/python trigger_ias.py                # list the scripts
   .venv/Scripts/python trigger_ias.py cot_one_shot   # run one

Scripts that call a model need the Docker stack up with ``qwen3.5-4b`` loaded
(``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``).

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

   experimentation.illustrated_agents_scripts
   experimentation.illustrated_agents_scripts.one_shot
   experimentation.illustrated_agents_scripts.cot_one_shot
   experimentation.illustrated_agents_scripts.zero_shot_cot
   experimentation.illustrated_agents_scripts.zero_shot_cot_raw
   experimentation.illustrated_agents_scripts.run_agent_trajectory
   experimentation.illustrated_agents_scripts.diff_ch1_ch2
   experimentation.illustrated_agents_scripts.diff_ch1_ch6
   trigger_ias
