"""Tests for tiny_agent.llm_client.embedding: the ``EmbeddingModel`` client, against the real embedding model.

These talk to EmbeddingGemma as deployed in Docker (the ``llama-embed`` container of ``deploy/llm-client``,
``node execute.mjs up llm-client``) at ``TINYAGENT_EMBEDDING_URL`` (default ``http://127.0.0.1:3201/v1``). There
is no fake: when nothing answers there, the tests that need the model are skipped with the command that starts it.
"""

import math
import os

import pytest

from tiny_agent.llm_client.embedding import EmbeddingModel

URL = os.environ.get("TINYAGENT_EMBEDDING_URL", "http://127.0.0.1:3201/v1")


@pytest.fixture(scope="module")
def embedder():
    """The real EmbeddingGemma client, or a skip when the container is not running."""
    model = EmbeddingModel("embeddinggemma-300m", base_url=URL)
    try:
        model.embed("ping")
    except RuntimeError as error:
        pytest.skip(f"no embedding model at {URL} ({error}); start it with: node execute.mjs up llm-client")
    return model


def cosine(a, b):
    """Cosine similarity of two vectors."""
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_embed_returns_a_768_number_vector_of_floats(embedder):
    vector = embedder.embed("Dolphins are amazing!")
    assert len(vector) == 768 and all(isinstance(x, float) for x in vector)


def test_the_same_text_gives_the_same_vector(embedder):
    assert cosine(embedder.embed("I love flamingos."), embedder.embed("I love flamingos.")) == pytest.approx(1.0, abs=1e-4)


def test_related_texts_are_closer_than_unrelated_ones(embedder):
    a = embedder.embed("I love flamingos.")
    b = embedder.embed("Dolphins use echolocation.")
    c = embedder.embed("Flamingos are pink birds.")
    assert cosine(a, c) > cosine(a, b)
    assert cosine(a, c) > cosine(b, c)


def test_a_trailing_slash_on_the_url_is_fine(embedder):
    assert len(EmbeddingModel("embeddinggemma-300m", base_url=URL + "/").embed("x")) == 768


def test_an_unreachable_server_becomes_a_runtime_error(dead_url):
    with pytest.raises(RuntimeError, match="Cannot reach the embedding model"):
        EmbeddingModel("embeddinggemma-300m", base_url=dead_url).embed("hi")
