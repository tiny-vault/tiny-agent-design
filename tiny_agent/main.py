"""Entry point for the TinyAgent demo: two tasks on one agent, to show its memory.

Role in the agentic system: the "user interface" layer. It creates the agent,
hands it two tasks, and shows the result. It contains no agent logic itself, so
the agent can later be driven from a CLI, a chat loop, or a test just by
changing this file.

Harness: `TinyAgent` (in tiny_agent/agent.py) is the harness, the non-model code that
turns an LLM into an agent. This file only starts it: `main()` creates the
harness and calls `run(task)`, which is where the loop, memory, tools and
planning are meant to live as the phases add them (today `run` makes one
model call, with the conversation so far from `memory`). Keeping the entry point separate means the harness can be
reused unchanged by any front end.

Settings come from ``.env`` or the environment: ``TINYAGENT_MODEL``
(default ``gemma4-e4b``), ``TINYAGENT_LLM_URL`` (default the Docker gateway) and
an optional API key. Start the stack and load the model first
(``node execute.mjs up``); the reply varies from run to run.

The demo tells the agent the authors' names, then asks for them in a second run. The second call
sends the whole conversation, so the model can answer; without memory it could not.

Set ``TINYAGENT_TRIM_MEMORY=1`` to give the agent a ``TrimmingMemory`` (it keeps only the last two
turns) instead of the plain ``Memory``. The two tasks here fit in the window, so the demo's answer does not
change; the flag matters in a longer conversation.

Set ``TINYAGENT_SUMMARY_MEMORY=1`` to give it a ``SummarizationMemory`` instead: after each reply the model
rewrites a running summary and the summary replaces the history. That costs one extra model call per task,
and if both flags are set the summary wins.

Set ``TINYAGENT_RAG_MEMORY=1`` to give it a ``RAGMemory`` instead: each user message gets the three most relevant
``DOCUMENTS`` put in front of it, found by an embedding model (``TINYAGENT_EMBEDDING_MODEL``, default
``embeddinggemma-300m``) at ``TINYAGENT_EMBEDDING_URL`` (default ``http://127.0.0.1:3201/v1``, the
``llama-embed`` container of ``deploy/llm-client``). It wins over the other two flags.

Set ``TINYAGENT_BANK_MEMORY=1`` to give it a ``MemoryBank`` instead: after each reply the model writes an event summary
and updates a user portrait, and each user message gets the summaries it recalls (and the portrait) put in front of
it; unused summaries fade over time. It needs the same embedding model as the RAG flag and costs two extra chat
calls and, for a nonempty summary, one embedding request per completed turn.
Retrieval adds an embedding request when retained summaries exist.
It wins over all the other flags.

Set ``TINYAGENT_TOOLS=1`` to bind the built-in tools (``multiply``) to the agent, described to the model in its prompt, or
``TINYAGENT_NATIVE_TOOLS=1`` to bind them for a model that learned tool calling (the definitions go in the request's
``tools`` field instead). The native flag wins. With neither, the agent has no tools.

Run it with ``python -m tiny_agent.main``. It logs to ``main.log`` and prints the second answer.
"""

from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.embedding import EmbeddingModel
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import BaseMemory, Memory
from tiny_agent.components.memory.memory_bank import MemoryBank
from tiny_agent.components.memory.rag import RAGMemory
from tiny_agent.components.memory.summarization import SummarizationMemory
from tiny_agent.components.memory.trimming import TrimmingMemory
from tiny_agent.components.tools.default import tools_from_settings
from tiny_agent.config.settings import Settings
from tiny_agent.logging import log as logs

# The first task puts the names into memory; the second can only be answered from it.
GREETING = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."
QUESTION = "Hi! What are our names?"

# What ``RAGMemory`` can retrieve from when TINYAGENT_RAG_MEMORY is on (from the book's example).
DOCUMENTS = [
    "Sarah works as a marine biologist studying coral reefs.",
    "Sarah lives in Lisbon, Portugal.",
    "Sarah's favorite hobby is rock climbing.",
    "Sarah favorite animal is flamingos.",
    "Sarah speaks fluent Spanish and Portuguese.",
    "Ilse is a software engineer at a renewable energy startup.",
    "Ilse lives in Amsterdam, the Netherlands.",
    "Ilse plays the cello in a local string quartet.",
    "Ilse's favorite author is Brandon Sanderson.",
    "Ilse's favorite animal is dolphins.",
]


def memory_from_env(llm: LLM | None = None, settings: Settings | None = None) -> BaseMemory:
    """Pick the agent's memory: banking, retrieving, summarizing, trimming or plain, depending on the environment.

    ``TINYAGENT_BANK_MEMORY`` gives a ``MemoryBank``, ``TINYAGENT_RAG_MEMORY`` a ``RAGMemory`` over ``DOCUMENTS``, ``TINYAGENT_SUMMARY_MEMORY`` a
    ``SummarizationMemory``, ``TINYAGENT_TRIM_MEMORY`` a ``TrimmingMemory``, and with none it is the plain ``Memory``. A flag is off unless it is ``1``, ``true``, ``yes`` or ``on``
    (any case), so the default behaviour is unchanged. Trimming keeps only the last two turns, so the agent
    forgets what was said before that; summarizing keeps a model-written summary instead. If both flags are on,
    the summary wins. Retrieval wins over both; its embedding model is ``TINYAGENT_EMBEDDING_MODEL`` (default
    ``embeddinggemma-300m``) at ``TINYAGENT_EMBEDDING_URL``, with the key of ``llm``, and building the memory embeds every
    document. The bank wins over all the others and uses the same embedding model.

    Args:
        settings: Explicit configuration; ``None`` reads ``Settings.from_env()``.
        llm: The model that writes the summaries, or whose key the embedding model uses. Needed only
            when ``TINYAGENT_SUMMARY_MEMORY`` ``TINYAGENT_RAG_MEMORY`` or ``TINYAGENT_BANK_MEMORY`` is on.

    Returns:
        The selected memory; retrieval embeds documents at construction.

    Raises:
        ValueError: ``TINYAGENT_SUMMARY_MEMORY``, ``TINYAGENT_RAG_MEMORY`` or ``TINYAGENT_BANK_MEMORY`` is on but no ``llm`` was given.
    """
    settings = settings or Settings.from_env()

    def embedding_model_for(llm: LLM) -> EmbeddingModel:
        """Share connection settings and the chat client's key with retrieval."""
        return EmbeddingModel(
            model=settings.embedding_model,
            base_url=settings.embedding_url,
            api_key=llm.api_key,
        )

    if settings.memory == "bank":
        if llm is None:
            raise ValueError("TINYAGENT_BANK_MEMORY needs an llm to write the summaries and the embedding model's key")
        return MemoryBank(llm, embedding_model_for(llm))
    if settings.memory == "rag":
        if llm is None:
            raise ValueError("TINYAGENT_RAG_MEMORY needs an llm for the embedding model's key")
        return RAGMemory(embedding_model_for(llm), DOCUMENTS)
    if settings.memory == "summary":
        if llm is None:
            raise ValueError("TINYAGENT_SUMMARY_MEMORY needs an llm to write the summaries")
        return SummarizationMemory(llm)
    return TrimmingMemory() if settings.memory == "trim" else Memory()


def main() -> None:
    """Build the agent, run the greeting then the question, and print the second answer.

    Role: composition root. Constructs the agent (and, in later steps,
    configures its LLM, memory, tools and planner) before calling `run`.
    """
    log = logs.setup("main")  # writes to logs/main.log (see tiny_agent/logging/log.py)
    # The same settings as the tiny-agent CLI: TINYAGENT_MODEL, TINYAGENT_LLM_URL, TINYAGENT_API_KEY(_FILE).
    settings = Settings.from_env()
    api_key = settings.resolve_api_key()  # TINYAGENT_API_KEY_FILE or TINYAGENT_API_KEY
    key_setting: dict[str, Any] = {"api_key": api_key} if api_key else {}
    llm = LLM(model=settings.model or "gemma4-e4b", base_url=settings.llm_url, **key_setting)
    agent = TinyAgent(llm, memory_from_env(llm, settings), tools_from_settings(settings))
    for task in (GREETING, QUESTION):
        log.info("running task: %s", task)
        result = agent.run(task)
        log.info("result: %s", result)
    print(result)


if __name__ == "__main__":
    main()
