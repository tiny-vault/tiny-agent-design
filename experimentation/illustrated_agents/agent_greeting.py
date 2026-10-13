"""Run the phase 2 ``TinyAgent`` once on the authors' greeting and print its reply.

**What it is.** In phase 1 ``TinyAgent`` only echoed its task. Phase 2 gives it an ``llm``: ``run`` now sends the task to the model as one user message and returns the model's text. That is one LLM call, with no memory, tools or planning yet.

**How it works here.** ``agent`` is a ``TinyAgent`` built with ``llm``; ``run(authors_greeting)`` sends "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'." and the script prints the reply. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None`` (the server's default, like the book's call), standing in for the book's ``gemma3:12b``, which the stack does not serve.

**What to expect.** A friendly reply that greets Maarten and Jay; the wording varies between runs. Ask the agent their names in a second ``run`` and it cannot know them, because nothing is remembered between runs.

Compare ``agent_without_memory``, which does that second run, and ``run_agent_trajectory``, which prints the same agent's trajectory.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM, TinyAgent

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's call: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

# The book's listing wraps this string across two lines; joined here
authors_greeting = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."

if __name__ == "__main__":
    # Run a query (needs the Docker stack with qwen3.5-4b loaded)
    agent = TinyAgent(llm=llm)
    response = agent.run(authors_greeting)
    print(response)
