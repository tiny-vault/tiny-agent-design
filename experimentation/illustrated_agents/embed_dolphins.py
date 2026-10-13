"""Embed one sentence with EmbeddingGemma and look at the vector the book prints as ``output``.

**What it shows.** The book's test cell: create the embedding model and call ``embed`` on "Dolphins are amazing!". The cell ends with the bare name ``output``, which in a notebook displays the whole vector; here ``show`` prints its length and first values instead, because 768 numbers fill a screen.

**How it works here.** ``EmbeddingModel`` is the same small ``urllib`` client as in ``embedding_model`` (copied, since scripts do not import each other). ``embedding_model`` uses the stack's name ``embeddinggemma-300m`` rather than the book's ``embeddinggemma`` (the Ollama tag); ``output`` is the vector for ``text``.

**What to expect.** 768 floats, usually between about -1 and 1; the values depend on the model, not on this script. The server must have ``embeddinggemma-300m`` loaded (it unloads the chat model). Compare ``embedding_model``, which ranks three texts by cosine similarity.
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


# Initialize EmbeddingGemma (served by the Docker stack, not Ollama)
embedding_model = EmbeddingModel(model="embeddinggemma-300m")
text = "Dolphins are amazing!"


def show(vector):
    """Print the vector's size and its first and last values (the notebook would print all of them).

    Args:
        vector: The embedding returned by ``embed``.
    """
    print(f"TEXT    {text}")
    print(f"VECTOR  {len(vector)} numbers")
    print(f"  first {[round(x, 4) for x in vector[:5]]}")
    print(f"  last  {[round(x, 4) for x in vector[-5:]]}")


if __name__ == "__main__":
    # Test the embedding model (needs the Docker stack with embeddinggemma-300m loaded)
    output = embedding_model.embed(text)
    show(output)
