"""The phase 2 ``TinyAgent`` has no memory: tell it the authors' names, then ask for them in a second run.

**What it shows.** An LLM only sees the messages it is sent. The phase 2 agent sends just the current task, so each ``run`` starts from nothing. Memory (phase 4) is what carries earlier turns forward.

**How it works here.** ``agent`` first runs ``authors_greeting`` ("Hi! We are Maarten and Jay, ..."), then ``names_question`` ("Hi! What are our names?"). The second call sends only the question, never the greeting, so the model has no way to know the names.

**What to expect.** The model says it doesn't know your names (it may guess or ask for them); wording varies between runs. A correct guess would be luck, not memory. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None`` (the server's default, like the book's call), standing in for the book's ``gemma3:12b``.

Compare ``agent_greeting``, which runs only the first turn.
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

# The follow-up: the names were only in the previous run
names_question = "Hi! What are our names?"

if __name__ == "__main__":
    # Two runs on the same agent (needs the Docker stack with qwen3.5-4b loaded)
    agent = TinyAgent(llm=llm)
    response = agent.run(authors_greeting)
    print(f"{authors_greeting}\n-> {response}\n")

    # The names were only in the previous run, which this call does not include
    response = agent.run(names_question)
    print(f"{names_question}\n-> {response}")
