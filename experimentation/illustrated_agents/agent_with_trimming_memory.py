"""The phase 4 ``TinyAgent`` with a trimming memory: four tasks, and only the last two turns are remembered.

**What it shows.** ``Memory`` keeps every message and resends all of it on every call, so a long chat eventually
passes the model's context window. ``TrimmingMemory`` is the simplest fix: after each ``add`` it keeps the system
message(s) and only the last four other messages, which is the last two turns (a user message and the reply,
twice). The agent is the same ``TinyAgent``; only the ``memory`` it is given changes, which is the point of the
memory slot.

**How it works here.** ``memory`` is a ``TrimmingMemory`` and ``agent`` a ``TinyAgent`` built with ``llm`` and
that ``memory``. The four ``tasks`` run one after another on that agent: the book introduces itself, asks about
flamingos, then two arithmetic questions. After each ``run`` the script prints what the memory holds
(``show_memory``), so you can watch the window slide: it grows until it has four messages, then the oldest ones
drop out. By the fourth task the first two tasks are no longer in what the model is sent.

**What to expect.** The memory listing is deterministic (roles in order, never more than four messages); the
replies are not, so their wording varies between runs. The trim happens when a user message is added, so at the
moment of the third and fourth ``run`` the model is sent a window that starts with an assistant message; the
local model used here accepts that, but a stricter chat template might not. Nothing is summarised: what leaves
the window is forgotten. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None`` (the server's
default). Compare ``agent_with_memory``, which keeps every message, and the repo's own ``TrimmingMemory`` in
``tiny_agent/components/memory/trimming.py``.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch4 import TinyAgent, TrimmingMemory

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's calls: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

# The book's four interactions. The first two give the memory something to forget; the last two are small
# questions whose answers do not depend on the earlier turns.
tasks = [
    "Hi! I'm reading 'An Illustrated Guide to AI Agents'.",
    "There are many flamingos in this book, why?",
    "What is 1+1?",
    "What is 2+2?",
]


def show_memory(memory):
    """Print what the memory holds, one line per message: the role and the start of the text.

    Args:
        memory: A ``Memory`` or ``TrimmingMemory``.
    """
    # Only the roles and the first words: enough to see which messages are still there.
    for message in memory.get_messages():
        print(f"    {message['role']:<9} {message['content'][:60]!r}")


if __name__ == "__main__":
    # One agent, four runs (needs the Docker stack with qwen3.5-4b loaded)
    memory = TrimmingMemory()
    agent = TinyAgent(llm=llm, memory=memory)
    for i, task in enumerate(tasks, 1):
        response = agent.run(task)
        print(f"RUN {i}/{len(tasks)}: {task}\n-> {response}")
        # After the reply is stored, so this is the window the next run starts from
        print(f"  memory now holds {len(memory.get_messages())} message(s):")
        show_memory(memory)
        print()
