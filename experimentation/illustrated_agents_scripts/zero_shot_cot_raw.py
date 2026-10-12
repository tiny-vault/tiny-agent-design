"""Zero-shot chain-of-thought prompting, printing the complete ``Response`` object.

Technique:
    Identical to :mod:`~experimentation.illustrated_agents_scripts.zero_shot_cot`: the question is sent once with "Let's think step by step." appended, and no worked example. The only difference is the output: this script prints the ``Response`` with plain ``print``, so you see the dataclass repr exactly as the book's notebook shows it.

Messages sent:
    1. ``user``: :data:`query` followed by " Let's think step by step."

Expected output:
    One line such as ``Response(content='Here is the step-by-step ...\\n\\n1.  ...You can see **11** penguins.', reasoning=None, tool_call=None, metadata={'model': 'qwen3.5-4b', 'prompt_tokens': 53, 'completion_tokens': 149})``. Newlines in ``content`` appear escaped as ``\\n``. The fields are:

    * ``content``: the answer text, here the step-by-step reasoning ending in 11.
    * ``reasoning``: separate thinking text when the server sends it and thinking is on; ``None`` here.
    * ``tool_call``: the tool the model asked to run; ``None`` here, since no tools are offered.
    * ``metadata``: ``model``, ``prompt_tokens`` and ``completion_tokens`` for the call.

Run:
    ``.venv/Scripts/python trigger_ias.py zero_shot_cot_raw`` from the repo root.

Requires:
    The Docker stack up with ``qwen3.5-4b`` loaded (``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``) and the secret file ``experimentation/chat-app/secrets/llama_api_key``, which ``up`` creates. Importing the module reads the secret but makes no network call.
"""

from pathlib import Path

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

# Messages with zero-shot Chain-of-Thought prompting
messages = [{"role": "user", "content": query + " Let's think step by step."}]
"""A single ``user`` message, :data:`query` plus the zero-shot CoT trigger, with no example turns."""

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    # Print the whole Response object (all fields, as its repr)
    print(llm.generate(messages))
