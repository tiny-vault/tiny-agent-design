"""Zero-shot chain-of-thought: no example, just "Let's think step by step." (Kojima et al., 2022).

One ``user`` message: the penguin ``query`` (answer 11) plus the trigger. Shortest prompt, longest reply: a numbered breakdown ending in 11. ``show`` prints each ``Response`` field on its own line. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``. ``zero_shot_cot_raw`` prints the raw ``Response`` instead.
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


# Messages with zero-shot Chain-of-Thought prompting
messages = [{"role": "user", "content": query + " Let's think step by step."}]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
