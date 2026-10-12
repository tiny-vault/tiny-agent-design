"""One-shot chain-of-thought: the solved example shows its working, so the model does too.

The chat is the flamingo example (``prompt_example``), its step-by-step answer (``prompt_answer``, 6 - 2 = 4, 4 - 1 = 3), then the penguin ``query`` (answer 11). Expect 9 - 2 = 7, 7 + 4 = 11 in about 100 to 115 tokens. ``show`` prints each ``Response`` field on its own line. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``. Compare ``one_shot`` and ``zero_shot_cot``.
"""

from pathlib import Path
from dataclasses import fields

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

# The query we want to ask the model
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"


def show(response):
    for f in fields(response):
        value = getattr(response, f.name)
        print(f"content:\n{value}" if f.name == "content" and value else f"{f.name}: {value}")


# Example of one-shot chain-of-thought prompting
prompt_example = "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?"

prompt_answer = "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3."

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
