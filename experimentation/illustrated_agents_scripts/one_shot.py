"""One-shot prompting: show the model one solved example, then ask the real question.

Technique:
    One-shot prompting puts a single worked example in the conversation before the real query. The example is a fake earlier turn: a ``user`` message with a similar word problem and an ``assistant`` message with its answer. The model copies both the *task* and the *format* of that answer. Because the example answer here is the bare number ``"3"`` and the example question ends with "Be concise.", the model replies to the penguin question with a bare number too.

Messages sent:
    1. ``user``: the flamingo example (:data:`prompt_example`).
    2. ``assistant``: its answer, ``"3"`` (:data:`prompt_answer`).
    3. ``user``: the penguin question (:data:`query`).

Expected output:
    ``content`` is ``11`` (9 - 2 + 4), with no explanation. ``reasoning`` and ``tool_call`` are ``None``. A typical call uses about 85 prompt tokens and 3 completion tokens. Compare :mod:`~experimentation.illustrated_agents_scripts.cot_one_shot`, where the example answer shows its working and the model shows its working as well.

Run:
    ``.venv/Scripts/python trigger_ias.py one_shot`` from the repo root.

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

    The default ``print(response)`` shows the dataclass repr on one line, with newlines in ``content`` escaped as ``\\n``. This prints ``content`` as real text under a ``content:`` heading, then every other field as ``name: value``, in the dataclass's field order (``content``, ``reasoning``, ``tool_call``, ``metadata``).

    Args:
        response: The ``Response`` dataclass returned by ``LLM.generate``. Any dataclass works; only ``content`` gets special treatment.

    Returns:
        None. Output goes to stdout.

    Example:
        >>> show(Response(content="11", metadata={"model": "qwen3.5-4b"}))  # doctest: +SKIP
        content:
        11
        reasoning: None
        tool_call: None
        metadata: {'model': 'qwen3.5-4b'}
    """
    for f in fields(response):
        value = getattr(response, f.name)
        print(f"content:\n{value}" if f.name == "content" and value else f"{f.name}: {value}")


# Example of one-shot prompting
prompt_example = "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see? Be concise."
"""The example question, shaped like :data:`query`. "Be concise." pushes the model toward a short answer."""

prompt_answer = "3"
"""The example answer, just the number (6 - 2 - 1), with no working, so the model imitates a bare answer."""

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]
"""The chat sent to the model, in order the example question, the example answer and the real :data:`query`."""

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
