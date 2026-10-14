"""Cosine similarity: how alike two embeddings are, shared by ``RAGMemory`` and ``MemoryBank``."""


def cosine(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embeddings.

    Args:
        a: One embedding.
        b: Another embedding of the same length.

    Returns:
        1 for the same direction, near 0 for unrelated texts.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return dot / norm
