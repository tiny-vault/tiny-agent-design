"""One-shot chain-of-thought prompting: a solved example that shows its working.

Technique:
    Chain-of-thought (CoT) prompting gets a model to write intermediate reasoning steps before its answer, which makes multi-step arithmetic more reliable. The one-shot variant does this by example: the single worked example's answer spells out each step (6 - 2 = 4, then 4 - 1 = 3) instead of giving only the number. The model imitates that format and reasons step by step about the real question. Compare :mod:`~experimentation.illustrated_agents_scripts.one_shot` (same example question, bare answer) and :mod:`~experimentation.illustrated_agents_scripts.zero_shot_cot` (no example, only an instruction to think step by step).

Messages sent:
    1. ``user``: the flamingo example (:data:`prompt_example`).
    2. ``assistant``: its step-by-step answer (:data:`prompt_answer`).
    3. ``user``: the penguin question (:data:`query`).

Expected output:
    ``content`` walks through 9 - 2 = 7, then 7 + 4 = 11, and ends with 11. ``reasoning`` and ``tool_call`` are ``None``. A typical call uses about 121 prompt tokens and 100 to 115 completion tokens: more output than one-shot, in exchange for visible working.

Run:
    ``.venv/Scripts/python trigger_ias.py cot_one_shot`` from the repo root.

Requires:
    The Docker stack up with ``qwen3.5-4b`` loaded (``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``) and the secret file ``experimentation/chat-app/secrets/llama_api_key``, which ``up`` creates. Importing the module reads the secret but makes no network call.
"""

from pathlib import Path
from dataclasses import fields

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)
"""The model client, ``qwen3.5-4b`` through the chat-app gateway at ``http://localhost:3200/v1``.

``temperature=1`` keeps the book's setting, so wording can vary between runs; the answer should not. Building the client makes no network call; ``LLM.generate`` does.

:meta hide-value:
"""

# The query we want to ask the model
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"
"""The real question; the correct answer is 11 (9 seen, 2 leave sight, 4 arrive)."""


def show(response):
    """Print a ``Response`` one field per line.

    The default ``print(response)`` shows the dataclass repr on one line, with newlines in ``content`` escaped as ``\\n``. This prints ``content`` as real text under a ``content:`` heading (so the model's numbered steps are readable), then every other field as ``name: value``, in the dataclass's field order (``content``, ``reasoning``, ``tool_call``, ``metadata``).

    Args:
        response: The ``Response`` dataclass returned by ``LLM.generate``. Any dataclass works; only ``content`` gets special treatment.

    Returns:
        None. Output goes to stdout.
    """
    for f in fields(response):
        value = getattr(response, f.name)
        print(f"content:\n{value}" if f.name == "content" and value else f"{f.name}: {value}")


# Example of one-shot chain-of-thought prompting
prompt_example = "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?"
"""The example question, shaped like :data:`query`. Unlike in ``one_shot`` it does not ask for a concise answer."""

prompt_answer = "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3."
"""The example answer, written as a chain of thought (one arithmetic step per event, then the result). This is the format the model copies."""

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]
"""The chat sent to the model, in order the example question, the step-by-step example answer and the real :data:`query`."""

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
