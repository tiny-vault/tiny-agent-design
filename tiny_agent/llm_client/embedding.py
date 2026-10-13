"""The embedding client: turn a text into a vector with an OpenAI-compatible ``/v1/embeddings`` endpoint.

An embedding is a list of numbers that places a text in a space where texts with similar meanings sit close
together. Comparing two vectors (for example with cosine similarity) tells how related two texts are, which is
what retrieval-augmented memory uses to find the notes that match a question.

``EmbeddingModel`` is the counterpart of :class:`~tiny_agent.llm_client.llm.LLM` for embedding models such as EmbeddingGemma
(``embeddinggemma-300m`` on the Docker stack): it holds the connection settings and ``embed`` does one request.
Standard library only, like the rest of this package.

Limitations: one text per request, blocking, no retries. The model must be loaded on the server (on the Docker
stack that means ``LLAMA_MODELS_MAX`` of at least 2 if a chat model is loaded as well).
"""

import http.client
import json
import urllib.error
import urllib.request


class EmbeddingModel:
    """One embedding model on one server.

    Attributes:
        model: The model name the server knows, for example ``"embeddinggemma-300m"``.
        base_url: The API root, ending in ``/v1``. ``/embeddings`` is added to it.
        api_key: Sent as a bearer token. Local servers ignore it.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:3200/v1",  # the Docker gateway; Ollama is http://localhost:11434/v1
        api_key: str = "no_key",
    ):
        """Initialize the embedding model with the given model.

        Nothing is sent to the server here; the first request happens in :meth:`embed`.

        Args:
            model: The model name the server knows.
            base_url: The API root, ending in ``/v1``. Defaults to the Docker gateway.
            api_key: Bearer token sent with every request. The default is a placeholder that local servers ignore.
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector.

        Args:
            text: The text to embed.

        Returns:
            The embedding, one float per dimension (768 for EmbeddingGemma).

        Raises:
            RuntimeError: The server cannot be reached, answered with an error status or with something that is
                not an embedding. The message says which; the original exception is kept as ``__cause__``.
        """
        # POST to the OpenAI-compatible /embeddings endpoint
        try:
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/embeddings",
                data=json.dumps({"model": self.model, "input": text}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(request, timeout=120) as resp:
                response = json.loads(resp.read())
        except urllib.error.HTTPError as error:  # the server answered with an error status
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = detail[:500]
            raise RuntimeError(f"HTTP {error.code} from {self.base_url}: {detail}") from error
        except (OSError, http.client.HTTPException) as error:  # refused, reset, timed out
            raise RuntimeError(f"Cannot reach the embedding model at {self.base_url}: {error}") from error
        except ValueError as error:  # a malformed URL, or a reply that is not JSON
            raise RuntimeError(f"Bad URL or a reply that is not JSON ({self.base_url}): {error}") from error

        # Extract and return the embedding, after checking the reply really holds one
        try:
            return [float(x) for x in response["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Unexpected reply from the embedding model: {str(response)[:200]}") from error
