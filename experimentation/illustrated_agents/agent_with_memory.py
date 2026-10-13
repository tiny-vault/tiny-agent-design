"""The phase 4 ``TinyAgent`` has memory: tell it the authors' names, then ask for them in a second run.

**What it shows.** In phase 2 ``run`` sent only the current task to the model, so each call started from nothing. Phase 4 gives ``TinyAgent`` a ``memory``: ``run`` adds the task to it as a user message, sends the model the whole stored conversation, and adds the reply to it as an assistant message. The model still only sees the messages it is sent, but now they include every earlier turn. That is the whole difference from ``agent_without_memory``.

**How it works here.** ``agent`` is a ``TinyAgent`` built with ``llm`` and a fresh ``Memory`` (a list of messages, no trimming or summarising). It first runs ``authors_greeting`` ("Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."), then ``names_question`` ("Hi! What are our names?"). The second call sends the greeting, the first reply and the question, so the model can answer from the conversation.

**What to expect.** A friendly reply to the greeting, then an answer naming Maarten and Jay; wording varies between runs. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None`` (the server's default, like the book's call), standing in for the book's ``gemma3:12b``, which the stack does not serve. Compare ``agent_without_memory``, which runs the same two tasks without memory.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch4 import Memory, TinyAgent

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's call: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

# The book's listing wraps this string across two lines; joined here
authors_greeting = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."

# The follow-up: the names are only in the previous run, which memory carries forward
names_question = "Hi! What are our names?"

if __name__ == "__main__":
    # Two runs on the same agent and the same memory (needs the Docker stack with qwen3.5-4b loaded)
    memory = Memory()
    agent = TinyAgent(llm=llm, memory=memory)
    response = agent.run(authors_greeting)
    print(f"{authors_greeting}\n-> {response}\n")

    # Memory holds the greeting and its reply, so this call can answer from them
    response = agent.run(names_question)
    print(f"{names_question}\n-> {response}")
