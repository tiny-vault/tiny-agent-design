"""``EmbeddingModel``: turn text into a vector by calling the server's OpenAI-compatible ``/embeddings`` endpoint.

**What it is.** An embedding is a list of numbers that places a text in a space where similar meanings sit close together. ``EmbeddingModel`` is the small client for it: ``embed(text)`` POSTs the text to ``/embeddings`` and returns the vector. It uses only the standard library (``urllib``), like the book's version.

**How it works here.** ``embedding_model`` is an ``EmbeddingModel`` for ``embeddinggemma-300m`` (Google DeepMind's 308-million-parameter EmbeddingGemma, served by llama.cpp from ``models.ini``) at the gateway. The three ``texts`` are embedded one by one; the script prints each vector's length and first values, then the cosine similarity of ``query`` against each text (``cosine_similarity``), best match first.

**What to expect.** Vectors have 768 numbers. The text about penguins should score highest against ``query`` and the unrelated one lowest, but the exact scores are not fixed by this script. The server must have ``embeddinggemma-300m`` loaded (``LLAMA_MODELS_MAX=1``, so loading it unloads the chat model); a request while it is not loaded fails with an HTTP error. ``api_key`` is sent as a bearer token, which a server without a key ignores. Compare ``shot_prompting/zero_shot`` for the chat side of the same server.
"""

import json
import math
import urllib.request
from pathlib import Path

# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()


class EmbeddingModel:
    """Generate embeddings."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:3200/v1",
    ):
        """Initialize the embedding model with the given model.

        Args:
            model: Name of the embedding model in ``models.ini``.
            base_url: The OpenAI-compatible server, ending in ``/v1``.
        """
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector.

        Args:
            text: The text to embed.

        Returns:
            The embedding, one float per dimension.
        """
        # POST to the OpenAI-compatible /embeddings endpoint
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read())

        # Extract and return the embedding
        return response["data"][0]["embedding"]


# EmbeddingGemma (308 M parameters), served by the Docker stack (llama.cpp behind the gateway), not Ollama.
embedding_model = EmbeddingModel(model="embeddinggemma-300m")

query = "Which birds cannot fly?"
texts = [
    "Penguins are flightless birds that swim in cold southern waters.",
    "A flamingo stands on one leg in shallow water.",
    "The quarterly invoice is due at the end of the month.",
]


def cosine_similarity(a, b):
    """Return the cosine of the angle between two vectors: 1 means the same direction, 0 unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def show(vectors):
    """Print each text's vector size and first values, then the texts ranked by similarity to ``query``.

    Args:
        vectors: The embedding of ``query`` followed by those of ``texts``.
    """
    query_vector, text_vectors = vectors[0], vectors[1:]
    print(f"QUERY  {query}")
    for text, vector in zip(texts, text_vectors):
        print(f"  {len(vector)} numbers, starting {[round(x, 3) for x in vector[:4]]}  <- {text}")
    print("RANKED by cosine similarity to the query")
    ranked = sorted(zip(texts, text_vectors), key=lambda pair: cosine_similarity(query_vector, pair[1]), reverse=True)
    for text, vector in ranked:
        print(f"  {cosine_similarity(query_vector, vector):.3f}  {text}")


if __name__ == "__main__":
    # Needs the Docker stack with embeddinggemma-300m loaded
    show([embedding_model.embed(t) for t in [query, *texts]])
