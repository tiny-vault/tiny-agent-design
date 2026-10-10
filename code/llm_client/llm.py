"""The "brain": a client for a chat model behind an OpenAI-compatible HTTP API.

Step 1 of the harness. An LLM only maps text to text, so this module's one job
is to send it a conversation and hand back what it said as a
:class:`~llm_client.response.Response`: the answer text, the model's
reasoning if it sent any, and the tool it asked to run if it wants one.

It speaks the ``/v1/chat/completions`` dialect that most local and hosted model
servers share, so the same code works against:

* the chat app's Docker gateway (llama.cpp behind it), the default,
  ``http://localhost:3200/v1``. Model names are the ones in
  ``experimentation/chat-app/models/models.ini``, for example ``gemma4-e4b``.
* Ollama, ``http://localhost:11434/v1``. Model names are Ollama tags, for
  example ``gemma4:e4b-it-qat``.

The request follows the example in *An Illustrated Guide to AI Agents* by
Maarten Grootendorst and Jay Alammar (O'Reilly): POST ``model`` and ``messages``
as JSON, then read ``result["choices"][0]``.

Only the standard library is used, so TinyAgent keeps zero runtime dependencies.

Example:
    >>> llm = LLM(model="gemma4-e4b")                            # doctest: +SKIP
    >>> llm.generate([{"role": "user", "content": "Hi!"}]).content  # doctest: +SKIP
    'Hello! How can I help?'
"""

import http.client
import json
import urllib.error
import urllib.request
import warnings

from llm_client import check
from llm_client.response import Response

class LLM:
    """One chat model on one server.

    An ``LLM`` holds the connection settings and turns a list of messages into a
    :class:`~llm_client.response.Response` with :meth:`generate`. It keeps
    no conversation state: the caller passes the full message history every time,
    which is what lets the harness (and later its memory) decide what the model
    sees.

    Attributes:
        model: The model name the server knows. On the Docker gateway this is a
            name from ``models.ini`` such as ``"gemma4-e4b"``; on Ollama it is a
            tag such as ``"gemma4:e4b-it-qat"``.
        base_url: The API root, ending in ``/v1``. ``/chat/completions`` is
            added to it.
        api_key: Sent as a bearer token. Local servers ignore it, hosted ones
            require a real key.
        think: Whether the model may reason before answering. ``False`` asks it
            not to (faster, shorter); ``True`` lets it, and its reasoning is
            returned in ``Response.reasoning``.

    Note:
        On the Docker gateway only one model is loaded at a time, and a model
        that is not loaded is refused. :meth:`generate` checks first and says how
        to load it (``node execute.mjs load <model>``).

    Example:
        >>> llm = LLM(model="gemma4-e4b", think=True)  # doctest: +SKIP
        >>> reply = llm.generate([{"role": "user", "content": "2 + 2?"}])  # doctest: +SKIP
        >>> reply.content, reply.metadata["completion_tokens"]  # doctest: +SKIP
        ('4', 312)
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:3200/v1",  # the Docker gateway; Ollama is http://localhost:11434/v1
        api_key: str = "no_key",
        think: bool = False,
    ):
        """Initialize the LLM with the given model.

        Nothing is sent to the server here; the first request happens in
        :meth:`generate`.

        Args:
            model: The model name the server knows, for example ``"gemma4-e4b"``
                (Docker gateway) or ``"gemma4:e4b-it-qat"`` (Ollama).
            base_url: The API root, ending in ``/v1``. Defaults to the Docker
                gateway. Pass ``"http://localhost:11434/v1"`` for Ollama.
            api_key: Bearer token sent with every request. The default is a
                placeholder that local servers ignore.
            think: ``False`` (default) sends ``reasoning_effort: "none"`` so the
                model answers without a long reasoning phase. ``True`` leaves it
                free to think; the thinking comes back in ``Response.reasoning``.
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think

    def generate(
        self, messages: list[dict], tools: list | None = None
    ) -> Response:
        """Generate a response from the LLM given a list of messages.

        Role: the one place the harness talks to the model. It asks the server
        which models are loaded, sends the conversation, and converts the JSON
        reply into a :class:`~llm_client.response.Response`.

        The call blocks until the model has finished answering. It gives up after
        600 seconds, which is long enough for a slow local model to reason.

        Args:
            messages: The conversation so far, in OpenAI format:
                ``[{"role": "system", "content": "..."}, {"role": "user",
                "content": "..."}]``. Roles are ``system``, ``user``,
                ``assistant`` and ``tool``; roles must alternate sensibly
                (some chat templates reject two ``user`` messages in a row).
            tools: Optional descriptions of tools the model may call, in the
                OpenAI function format
                (``{"type": "function", "function": {"name", "description",
                "parameters"}}``). Leave ``None`` for a plain chat.

        Returns:
            A ``Response``. ``content`` is the answer text, or ``None`` when the
            model only asked for a tool. ``reasoning`` is its thinking, read from
            either ``reasoning`` (Ollama) or ``reasoning_content`` (llama.cpp).
            ``tool_call`` is the first tool call as the server sent it
            (``{"id", "type", "function": {"name", "arguments"}}``, with
            ``arguments`` still a JSON string); any further calls in the same
            reply are dropped. ``metadata`` holds ``model``, ``prompt_tokens``
            and ``completion_tokens`` (the counts are ``None`` if the server
            does not report usage).

        Raises:
            ValueError: ``messages`` is empty.
            RuntimeError: Anything went wrong talking to the model, and the
                message says what:

                * the server is a router that reports this model as ``unloaded``
                  or ``loading`` (raised before any chat request is sent, with
                  the command that loads it);
                * the router does not offer this model and no model at all is
                  loaded (raised with the list of available models);
                * the server cannot be reached, refused, reset the connection or
                  timed out;
                * the server answered with an error status (unknown model, bad
                  request, server error), with the server's own message;
                * the URL is malformed, or the reply is not JSON;
                * the reply is JSON but not a chat completion (no ``choices``).

                The original exception is kept as ``__cause__``.

        Note:
            Each call first makes one extra request to ``/models``. On a router
            (the Docker gateway) this decides what happens next:

            * the model is loaded: it is used;
            * the model is offered but ``unloaded`` or ``loading``: ``RuntimeError``;
            * the model is **not offered** at all: the first model that *is* loaded
              is used instead, and a ``UserWarning`` says so. The substitution is
              for this call only; ``self.model`` is unchanged, and
              ``Response.metadata["model"]`` reports the model that really answered.

            A server that reports no status (such as Ollama), or that cannot be
            reached, is never blocked or redirected by this check.
        """
        # Ask the server which models it has, and which are loaded. (A server that is
        # unreachable, or reports no status such as Ollama, is not blocked here.)
        model = self.model
        statuses = check.model_statuses(self.base_url)
        if statuses is not None:
            status = statuses.get(self.model)
            loaded = [name for name, s in statuses.items() if s == "loaded"]
            if status in ("unloaded", "loading"):
                # Offered but not loaded: stop, and say how to load it.
                raise RuntimeError(
                    f"Model '{self.model}' is {status} (loaded now: {', '.join(loaded) or 'none'}). "
                    f"Load it first: node execute.mjs load {self.model}"
                )
            if status is None and any(s in ("loaded", "loading", "unloaded") for s in statuses.values()):
                # Not offered by this router: use a model that is loaded instead.
                if not loaded:
                    raise RuntimeError(
                        f"Model '{self.model}' is not offered and no model is loaded. "
                        f"Available: {', '.join(statuses)}. Load one first: node execute.mjs load <model>"
                    )
                model = loaded[0]
                warnings.warn(
                    f"Model '{self.model}' is not offered by {self.base_url}; using the loaded model '{model}'.",
                    stacklevel=2,
                )

        if not messages:
            raise ValueError("messages must not be empty")

        # Build the request body
        body = {
            "model": model,
            "messages": messages,
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"

        # POST to the OpenAI-compatible /chat/completions endpoint. Every way this can
        # fail becomes a RuntimeError that says what happened, so callers handle one type.
        try:
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            # 600 s: long enough for a slow local model to reason, short enough not to hang forever.
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:  # the server answered with an error status
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = detail[:500]
            raise RuntimeError(f"HTTP {error.code} from {self.base_url}: {detail}") from error
        except (OSError, http.client.HTTPException) as error:  # refused, reset, timed out
            raise RuntimeError(f"Cannot reach the model at {self.base_url}: {error}") from error
        except ValueError as error:  # a malformed URL, or a reply that is not JSON
            raise RuntimeError(f"Bad URL or a reply that is not JSON ({self.base_url}): {error}") from error

        # Extract message, tool_call, and metadata
        try:
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected reply from the model server: {str(data)[:200]}") from error
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        usage = data.get("usage") or {}  # some servers omit usage
        metadata = {
            "model": data.get("model", self.model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            # Ollama calls it "reasoning"; llama.cpp calls it "reasoning_content".
            reasoning=message.get("reasoning") or message.get("reasoning_content"),
            tool_call=tool_call,
            metadata=metadata,
        )
