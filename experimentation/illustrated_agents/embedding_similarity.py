"""Cosine similarity of embeddings: "I love flamingos." is closer to a flamingo sentence than to a dolphin one.

**What it shows.** Two texts about similar things get vectors that point in similar directions. Cosine similarity measures that angle: the dot product of two vectors divided by the product of their lengths, 1 for the same direction, near 0 for unrelated texts.

**How it works here.** ``embedding_a``, ``embedding_b`` and ``embedding_c`` are the vectors for "I love flamingos.", "Dolphins use echolocation." and "Flamingos are pink birds.", made by ``embedding_model`` (``EmbeddingModel``, copied from ``embedding_model`` because scripts do not import each other; it uses the stack's name ``embeddinggemma-300m``). The script spells out the arithmetic as the book does, step by step, for A against B (``similarity_ab``), A against C (``similarity_ac``) and B against C (``similarity_bc``), and prints all three.

**What to expect.** ``similarity_ac`` should be clearly larger than ``similarity_ab`` and ``similarity_bc``, since A and C are both about flamingos; the exact values come from the model. The server must have ``embeddinggemma-300m`` loaded (it unloads the chat model). Compare ``embedding_model``, which ranks texts with a ``cosine_similarity`` function.
"""

import json
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


# EmbeddingGemma, served by the Docker stack (llama.cpp behind the gateway), not Ollama
embedding_model = EmbeddingModel(model="embeddinggemma-300m")


if __name__ == "__main__":
    # Needs the Docker stack with embeddinggemma-300m loaded
    # Create embeddings
    embedding_a = embedding_model.embed("I love flamingos.")
    embedding_b = embedding_model.embed("Dolphins use echolocation.")
    embedding_c = embedding_model.embed("Flamingos are pink birds.")

    # Calculate cosine similarity between A and B
    dot_ab = sum(x * y for x, y in zip(embedding_a, embedding_b))
    norm_a = sum(x * x for x in embedding_a) ** 0.5
    norm_b = sum(x * x for x in embedding_b) ** 0.5
    similarity_ab = dot_ab / (norm_a * norm_b)

    # Calculate cosine similarity between A and C
    dot_ac = sum(x * y for x, y in zip(embedding_a, embedding_c))
    norm_c = sum(x * x for x in embedding_c) ** 0.5
    similarity_ac = dot_ac / (norm_a * norm_c)

    print(f"Similarity between A and B: {similarity_ab}")
    print(f"Similarity between A and C: {similarity_ac}")

    # Calculate cosine similarity between B and C
    dot_bc = sum(x * y for x, y in zip(embedding_b, embedding_c))
    similarity_bc = dot_bc / (norm_b * norm_c)
    print(f"Similarity between B and C: {similarity_bc}")
