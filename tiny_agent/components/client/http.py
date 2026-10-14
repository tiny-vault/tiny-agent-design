"""The one HTTP call the model clients share: POST JSON to an OpenAI-compatible endpoint.

``LLM`` (chat) and ``EmbeddingModel`` (vectors) both POST JSON and turn every failure into one ``RuntimeError``.
That shared part lives here as a function, so neither client needs a base class. It uses ``httpx``.
"""

from typing import Any

import httpx


def post_json(base_url: str, api_key: str, path: str, body: dict[str, Any], timeout: float, what: str = "model") -> Any:
    """POST ``body`` as JSON to ``base_url + path`` and return the decoded JSON reply.

    Args:
        base_url: The API root, ending in ``/v1``.
        api_key: Sent as a bearer token. Local servers ignore it, hosted ones require a real key.
        path: The endpoint, for example ``"/chat/completions"``.
        body: The request body.
        timeout: Seconds to wait before giving up.
        what: What is being called (``"model"`` or ``"embedding model"``), used in the unreachable-server message.

    Raises:
        RuntimeError: The server cannot be reached, answered with an error status, or the URL is
            malformed or the reply is not JSON. The message says which; the original exception is kept
            as ``__cause__``.
    """
    url = f"{base_url.rstrip('/')}{path}"
    try:
        reply = httpx.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        reply.raise_for_status()
        return reply.json()
    except httpx.HTTPStatusError as error:  # the server answered with an error status
        try:
            detail = error.response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = error.response.text[:500]
        raise RuntimeError(f"HTTP {error.response.status_code} from {base_url}: {detail}") from error
    except (httpx.InvalidURL, httpx.UnsupportedProtocol, ValueError) as error:  # a malformed URL, or a reply that is not JSON
        raise RuntimeError(f"Bad URL or a reply that is not JSON ({base_url}): {error}") from error
    except httpx.TransportError as error:  # refused, reset, timed out
        raise RuntimeError(f"Cannot reach the {what} at {base_url}: {error}") from error
