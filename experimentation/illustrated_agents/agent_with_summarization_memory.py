"""The phase 4 ``TinyAgent`` with a summarizing memory: four tasks, and the memory is one running summary.

**What it shows.** ``Memory`` keeps every message and ``TrimmingMemory`` forgets all but the last two turns.
``SummarizationMemory`` keeps neither: after each assistant reply it asks the model to fold the new turn into a
summary, and the summary (one ``system`` message) replaces the whole history. The agent is the same ``TinyAgent``;
only the ``memory`` it is given changes.

**How it works here.** ``memory`` is a ``SummarizationMemory`` built with ``llm`` (the model that writes the
summaries) and ``agent`` a ``TinyAgent`` built with ``llm`` and that ``memory``. The four ``tasks`` run one after
another: the book introduces itself, asks about flamingos, then two arithmetic questions. After each ``run`` the
script prints what the memory holds (``show_memory``): a single ``system`` message with the summary, never more.

**What to expect.** The memory always holds exactly one message after a reply, but the summary's wording varies
between runs because the model writes it. Each task costs two model calls, the answer and the summary, so the
run is slower than with ``Memory``. Whatever the summary leaves out is forgotten, and the summary calls do not
appear in the agent's trajectory. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None`` (the
server's default). Compare ``agent_with_trimming_memory``, which keeps the last two turns verbatim, and the
repo's own ``SummarizationMemory`` in ``tiny_agent/memory/summarization.py``.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch4 import SummarizationMemory, TinyAgent

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's calls: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

# The book's four interactions. The first two give the summary something to remember; the last two are small
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
        memory: A ``Memory``, ``TrimmingMemory`` or ``SummarizationMemory``.
    """
    # The summary is the interesting text here, so show more of it than a chat message needs
    for message in memory.get_messages():
        print(f"    {message['role']:<9} {message['content'][:200]!r}")


if __name__ == "__main__":
    # One agent, four runs (needs the Docker stack with qwen3.5-4b loaded)
    memory = SummarizationMemory(llm)
    agent = TinyAgent(llm=llm, memory=memory)
    for i, task in enumerate(tasks, 1):
        response = agent.run(task)
        print(f"RUN {i}/{len(tasks)}: {task}\n-> {response}")
        # After the reply is stored (and summarised), so this is what the next run starts from
        print(f"  memory now holds {len(memory.get_messages())} message(s):")
        show_memory(memory)
        print()
