"""Cosine similarity: compare the directions of two embedding vectors.

Retrieval ranks documents or memories by how closely their vectors point in
the same direction. ``cosine`` divides the dot product by the product of the
vectors' Euclidean lengths. Normalization makes direction matter rather than
vector magnitude: ``[2, 0]`` and ``[5, 0]`` score 1, perpendicular directions
score 0, and opposite directions score -1.

``RAGMemory`` and ``MemoryBank`` use this score to rank candidates before
selecting the most similar. A high score is evidence of vector alignment,
not proof that a document is true or relevant to every interpretation of a
question. Meaning depends on the embedding model.

Inputs must be equal-length, nonzero vectors from a compatible embedding
space. This helper does not check those assumptions: ``zip`` truncates the
dot product for unequal lengths, and a zero norm raises ``ZeroDivisionError``.
It returns a number and makes no model calls or file writes.
"""


def cosine(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embeddings.

    Args:
        a: One embedding.
        b: Another embedding of the same length.

    Returns:
        1 for the same direction, 0 for perpendicular directions, and -1
        for opposite directions. Text relevance depends on the embeddings.

    Raises:
        ZeroDivisionError: Either vector has zero length (norm).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return dot / norm
