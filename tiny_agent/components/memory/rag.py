"""RAG memory: long-term knowledge from a set of documents, retrieved by meaning and added to each question.

WHAT IS IT? ``Memory`` keeps the conversation, but an agent often needs facts that were never said in it: notes
about a user, a product manual, a wiki. Putting all of them in every prompt does not scale. Retrieval-augmented
generation (RAG) keeps the documents outside the prompt and, for each question, fetches only the few that are
about it. ``RAGMemory`` does that inside the memory slot: the agent and the model are unchanged.

HOW IT WORKS: at construction every document is turned into a vector with the ``embedding_model``. When a
``user`` message is added, the message is embedded too, ``search`` ranks the documents by cosine similarity to
it, and the three best are put in front of the question as ``Context:`` before the message is stored. Other
roles are stored as they are. The model is then sent the history, whose last user message carries the context.

WHAT TO NOTICE: the stored user message is the augmented one, so the retrieved context stays in the history and
is resent on every later call (the history grows by the context each turn, and nothing is ever removed). The
question is embedded as it is, on its own, so a follow-up such as "and her hobby?" retrieves poorly. Every user
message costs one embedding request; the documents are embedded once, when the memory is created. There are
fewer than three results only if there are fewer than three documents. A document or question that embeds to
the zero vector would divide by zero; real embedding models do not produce one.

Switch it on in the demo with ``TINYAGENT_RAG_MEMORY=1`` (see ``main.memory_from_env``); the harness itself
takes whichever memory it is given.
"""

from typing import Any, override

from tiny_agent.components.client.embedding import EmbeddingModel
from tiny_agent.components.memory.memory import BaseMemory
from tiny_agent.components.memory.similarity import cosine


class RAGMemory(BaseMemory):
    """Long-term Memory with RAG.

    Role: a drop-in replacement for ``Memory`` in the harness's "context" slot, for agents that should answer
    from a set of documents. It has the same methods; only ``add`` differs, and it needs an embedding model.

    Attributes:
        messages: The conversation so far, as in ``Memory``; user messages include their retrieved context.
        embedding_model: The model that turns texts into vectors.
        documents: The texts that can be retrieved.
        embeddings: One vector per document, in the same order, computed once at construction.
    """

    def __init__(self, embedding_model: EmbeddingModel, documents: list[str]):
        """Start with an empty history and embed every document.

        Args:
            embedding_model: The model that embeds the documents and, later, each question.
            documents: The texts to retrieve from. Embedding them makes one request each, right here.
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    @override
    def add(self, role: str, content: str, tool_call: dict[str, Any] | None = None, **kwargs) -> None:
        """Add a message; a user message first gets the most relevant documents put in front of it.

        Args:
            role: Who said it: ``system``, ``user``, ``assistant`` or ``tool``.
            content: The message text. For ``user`` it becomes ``Context:`` plus the documents, then
                ``Question:`` plus this text.
            tool_call: The tool call the model made, if this message records one (stored as in ``Memory``).
            **kwargs: Accepted and ignored, as in ``Memory``.
        """
        # Augment user queries with retrieved context before storing
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        self._append(role, content, tool_call)

    def search(self, query: str) -> list[str]:
        """Return the top-k documents most similar to the query.

        Args:
            query: The text to look up (it is embedded with one request).

        Returns:
            Up to three documents, most similar first.
        """
        query_embed = self.embedding_model.embed(query)
        scores = [cosine(query_embed, embed) for embed in self.embeddings]
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self.documents[index] for index in ranked[:3]]
