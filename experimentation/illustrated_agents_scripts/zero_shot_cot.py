"""Zero-shot chain-of-thought prompting: no example, just "Let's think step by step."

Technique:
    Zero-shot CoT adds a reasoning trigger to the question itself instead of showing a worked example. Appending "Let's think step by step." (Kojima et al., 2022, "Large Language Models are Zero-Shot Reasoners") is enough for most instruction-tuned models to lay out intermediate steps before the answer. It costs no example tokens, so the prompt is the shortest of the three prompting scripts, but the model chooses its own format, so the reply is usually the longest. Compare :mod:`~experimentation.illustrated_agents_scripts.one_shot` and :mod:`~experimentation.illustrated_agents_scripts.cot_one_shot`.

Messages sent:
    1. ``user``: :data:`query` followed by " Let's think step by step."

Expected output:
    ``content`` is a numbered breakdown (9 - 2 = 7, then 7 + 4 = 11) ending with 11, printed field by field by :func:`show`. ``reasoning`` and ``tool_call`` are ``None``. A typical call uses about 53 prompt tokens and 145 to 170 completion tokens. :mod:`~experimentation.illustrated_agents_scripts.zero_shot_cot_raw` sends the same messages and prints the raw ``Response`` repr instead.

Run:
    ``.venv/Scripts/python trigger_ias.py zero_shot_cot`` from the repo root.

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

    The default ``print(response)`` shows the dataclass repr on one line, with newlines in ``content`` escaped as ``\\n``. This prints ``content`` as real text under a ``content:`` heading (so the model's steps are readable), then every other field as ``name: value``, in the dataclass's field order (``content``, ``reasoning``, ``tool_call``, ``metadata``).

    Args:
        response: The ``Response`` dataclass returned by ``LLM.generate``. Any dataclass works; only ``content`` gets special treatment.

    Returns:
        None. Output goes to stdout.
    """
    for f in fields(response):
        value = getattr(response, f.name)
        print(f"content:\n{value}" if f.name == "content" and value else f"{f.name}: {value}")


# Messages with zero-shot Chain-of-Thought prompting
messages = [{"role": "user", "content": query + " Let's think step by step."}]
"""A single ``user`` message, :data:`query` plus the zero-shot CoT trigger, with no example turns."""

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
