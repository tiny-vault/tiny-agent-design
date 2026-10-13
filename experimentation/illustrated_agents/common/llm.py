"""The ``LLM`` every script talks to: ``qwen3.5-4b`` on the Docker stack, not Ollama."""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Served by llama.cpp behind the gateway. The key is read at runtime from the git-ignored
# Docker secret (common/llm.py -> experimentation/chat-app/secrets) and never printed.
api_key = (Path(__file__).parents[2] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()


def make_llm(temperature=1):
    # temperature=1 (the book's setting) makes samples differ; 0 is greedy, e.g. for a judge
    return LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=temperature)


llm = make_llm()
