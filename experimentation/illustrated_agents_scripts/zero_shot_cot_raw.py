"""Same as ``zero_shot_cot``, but prints the raw ``Response`` repr, as the book's notebook does.

Expect one line: ``Response(content='...11...', reasoning=None, tool_call=None, metadata={...})``. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

# The query we want to ask the model
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"

# Messages with zero-shot Chain-of-Thought prompting
messages = [{"role": "user", "content": query + " Let's think step by step."}]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    # Print the whole Response object (all fields, as its repr)
    print(llm.generate(messages))
