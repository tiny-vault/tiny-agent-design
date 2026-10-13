"""Pieces shared by the scripts in ``experimentation/illustrated_agents``: the stack's ``LLM``, every prompt and plain-text printers.

``make_llm`` builds an ``LLM`` for ``qwen3.5-4b`` on the Docker stack (gateway ``http://localhost:3200/v1``, key read from ``chat-app/secrets/llama_api_key`` and never printed); ``llm`` is the default one at ``temperature=1``. ``common.prompts`` holds every prompt the scripts send (``penguin_query`` with its flamingo examples, ``step_by_step``, ``seating_query``, ``roman_to_int_query`` with ``ORM_PROMPT`` and ``PRM_PROMPT``, ``agent_query``). ``show`` prints each ``Response`` field on its own line, and ``print_diff`` prints a ``DiffViewer`` as a unified diff, since both viewers only render HTML inside Jupyter. Scripts import it as ``from common import ...``: run as a file, the script's folder is on ``sys.path``; ``trigger_ias.py`` and the docs build add that folder.
"""

from common.llm import api_key, llm, make_llm
from common.output import print_diff, show
from common.prompts import (ORM_PROMPT, PRM_PROMPT, agent_query, flamingo_answer, flamingo_cot_answer, flamingo_example,
                            penguin_query, roman_to_int_query, seating_query, step_by_step)

__all__ = ["api_key", "llm", "make_llm", "print_diff", "show",
           "ORM_PROMPT", "PRM_PROMPT", "agent_query", "flamingo_answer", "flamingo_cot_answer", "flamingo_example",
           "penguin_query", "roman_to_int_query", "seating_query", "step_by_step"]
