"""One query to a ``TinyAgent`` with a summarizing memory, and the summary it leaves behind.

**What it shows.** The first turn of a conversation with ``SummarizationMemory``: the user gives a fact (a name)
and asks a question in the same message. The agent answers, and the memory then replaces the exchange with a
summary, so the name should survive in the summary even though the messages are gone.

**How it works here.** ``memory`` is a ``SummarizationMemory`` built with ``llm`` and ``agent`` a ``TinyAgent``
built with ``llm`` and that ``memory``. ``query`` is sent with one ``agent.run``; ``response`` is the answer.
``show_memory`` then prints what the memory holds.

**What to expect.** An answer about why flamingos are pink (their food contains carotenoid pigments), and a memory
with exactly one ``system`` message, the summary, which should mention Sarah. The wording varies between runs
because the model writes the summary. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None`` (the
server's default). Compare ``agent_with_summarization_memory``, which runs four tasks and shows the summary
growing.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch4 import SummarizationMemory, TinyAgent

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
# No temperature, as in the book's calls: the server's default applies.
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

query = "Hi, my name is Sarah. Why are flamingos pink?"


def show_memory(memory):
    """Print what the memory holds, one line per message: the role and the text.

    Args:
        memory: A ``Memory``, ``TrimmingMemory`` or ``SummarizationMemory``.
    """
    for message in memory.get_messages():
        print(f"  {message['role']:<9} {message['content']!r}")


if __name__ == "__main__":
    # Create the agent, then run a query (needs the Docker stack with qwen3.5-4b loaded)
    memory = SummarizationMemory(llm=llm)
    agent = TinyAgent(llm=llm, memory=memory)
    response = agent.run(query)
    print(f"QUERY: {query}\n-> {response}\n")
    print(f"memory now holds {len(memory.get_messages())} message(s):")
    show_memory(memory)
