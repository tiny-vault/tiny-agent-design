"""The one HTTP call the model clients share: POST JSON to an OpenAI-compatible endpoint.

``LLM`` (chat) and ``EmbeddingModel`` (vectors) both POST JSON and turn every failure into one ``RuntimeError``.
That shared part lives here as a function, so neither client needs a base class. It uses ``httpx``.

The boundary has three stages: join the API root with the endpoint path, send
the body with a bearer-token header, then decode the response as JSON.
``post_json`` returns that decoded value without checking its application
schema; chat and embedding clients check their own expected shapes.

HTTP errors include the server's error message when available, otherwise a
bounded response excerpt. Transport failures, malformed URLs and invalid JSON
are wrapped as ``RuntimeError`` with the original exception as ``__cause__``.
Callers handle one transport-level failure type while retaining debugging
information. This does not cover arbitrary processing errors after decoding,
retry requests, keep history or stream results.

No request is made at import. Calling the helper blocks for the configured
httpx timeout and performs a network POST. A successful response can be any
JSON value, not necessarily a chat completion or embedding.
"""

from typing import Any

import httpx


def post_json(base_url: str, api_key: str, path: str, body: dict[str, Any], timeout: float, what: str = "model") -> Any:
    """POST ``body`` as JSON to ``base_url + path`` and return the decoded JSON reply.

    Args:
        base_url: The API root, ending in ``/v1``.
        api_key: Sent as a bearer token. Authentication requirements depend on the endpoint.
        path: The endpoint, for example ``"/chat/completions"``.
        body: The request body.
        timeout: Seconds to wait before giving up.
        what: What is being called (``"model"`` or ``"embedding model"``), used in the unreachable-server message.

    Returns:
        The decoded JSON value; endpoint-specific shape checks belong to callers.

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
