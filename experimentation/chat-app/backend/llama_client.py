"""Async client for llama-server's HTTP API (router mode), built on httpx.

What is it? ``llama-server`` (llama.cpp) runs language models on the GPU and
exposes them over HTTP. In *router mode* it knows several models but keeps only
one in memory at a time: asking for another one makes it unload the first and load
the new one, which takes tens of seconds (and minutes on the first download).
``LlamaClient`` is the backend's only way to talk to it.

Why a client class? The rest of the backend (``server.py``) should not know about
loading states, retries or timeouts. This file hides three things behind plain
async methods: listing models with a status (``loaded``, ``loading``, ``unloaded``),
sending one chat request, and ``load_model``, which asks the router to switch and
then polls until the model is really loaded (or fails, or times out).

The request shape for chat follows the example in *An Illustrated Guide to AI
Agents* by Maarten Grootendorst and Jay Alammar (O'Reilly): POST ``model`` and
``messages`` to ``/v1/chat/completions`` and read ``choices[0]``.

Failures become the exceptions below (``LlamaUnreachable``, ``LlamaError``,
``ModelNotFound``, ``LoadFailed``, ``LoadTimeout``), so ``server.py`` can map each
to one HTTP status without parsing httpx errors. This client belongs to the chat
app and is not used by ``code/tiny_agent``; that package has its own client.
"""

import asyncio
import time

import httpx

# Seconds an idle pooled connection is kept; must stay below llama-server's 5 s keep-alive timeout.
KEEPALIVE_EXPIRY = 2


class LlamaUnreachable(Exception):
    """llama-server could not be reached."""


class LlamaError(Exception):
    """llama-server answered with an error status.

    Attributes:
        status: The HTTP status code llama-server sent.
        message: The text from its error body (see ``_error_message``).
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class ModelNotFound(Exception):
    """The requested model is not one of the names llama-server offers."""


class LoadFailed(Exception):
    """The model started loading but ended up unloaded, or never started."""


class LoadTimeout(Exception):
    """The model did not reach "loaded" in time."""


class LlamaClient:
    """Talks to one llama-server. Create once, close with ``aclose``."""

    def __init__(self, base_url: str, *, api_key: str | None = None, timeout: float = 600,
                 load_timeout: float = 300, poll_interval: float = 1, start_grace: float = 15):
        """Create the HTTP connection pool; nothing is sent yet.

        Args:
            base_url: Where llama-server listens, for example ``http://llama:8080``.
            api_key: Sent as a bearer token on every request when llama-server was
                started with a key; ``None`` sends no ``Authorization`` header.
            timeout: Seconds to wait for a reply to one request. Long, because a chat
                answer from a local model can take minutes.
            load_timeout: Seconds ``load_model`` waits for a model to reach "loaded".
            poll_interval: Seconds between status checks while a model loads.
            start_grace: Seconds a model may still show "unloaded" after the load
                request before ``load_model`` decides it never started.
        """
        self.base_url = base_url.rstrip("/")
        # llama-server may require a key (--api-key-file). It is sent as a bearer token on every request.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # llama-server (cpp-httplib) closes a keep-alive connection after 5 s idle, the same as httpx's
        # default expiry, so a pooled connection could be reused just as the server dropped it: the
        # request then fails with "Server disconnected without sending a response" (a 502). Expiring
        # idle connections well before the server does means every reused connection is still open.
        self.http = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                      timeout=httpx.Timeout(timeout, connect=5),
                                      limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY))
        self.load_timeout = load_timeout
        self.poll_interval = poll_interval
        self.start_grace = start_grace

    async def aclose(self) -> None:
        """Close the connection pool; call once when the app shuts down."""
        await self.http.aclose()

    async def _json(self, method: str, path: str, **kwargs) -> dict:
        """Send one request and return its JSON body.

        Raises:
            LlamaUnreachable: The connection failed or timed out.
            LlamaError: llama-server answered with an error status.
        """
        try:
            response = await self.http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise LlamaUnreachable(str(error) or type(error).__name__) from error
        if response.is_error:
            raise LlamaError(response.status_code, _error_message(response))
        return response.json()

    async def is_up(self) -> bool:
        """True if llama-server answers ``/health`` with 200; never raises (used by the backend's own health check)."""
        try:
            return (await self.http.get("/health", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[dict]:
        """The model entries llama-server lists, without the ones it only found in its download cache.

        The router lists every model it knows: the ones in ``models.ini`` (source ``preset``) and,
        since llama.cpp keeps downloads in a cache, one extra entry for each cached download
        (source ``cache``). Only the preset ones are offered. A server that reports no source is
        not a router, so all of its entries are kept.
        """
        data = (await self._json("GET", "/v1/models")).get("data", [])
        return [m for m in data if m.get("source", "preset") == "preset"]

    async def model_statuses(self) -> dict[str, str]:
        """Return ``{model: status}`` with status loaded, loading or unloaded.

        A server without router mode reports no status; its models count as loaded.
        """
        data = await self.list_models()
        return {m["id"]: (m.get("status") or {}).get("value", "loaded") for m in data}

    async def chat(self, model: str, messages: list[dict], **options) -> dict:
        """Send a chat request and return the first choice of the reply."""
        result = await self._json("POST", "/v1/chat/completions",
                                  json={"model": model, "messages": messages, **options})
        return result["choices"][0]

    async def load_model(self, model: str) -> None:
        """Ask the router to load ``model`` and wait until it is "loaded".

        Raises:
            ModelNotFound: ``model`` is not offered by llama-server.
            LoadFailed: it went back to unloaded after loading started, or never
                started within ``start_grace`` seconds.
            LoadTimeout: still not loaded after ``load_timeout`` seconds.
            LlamaUnreachable: llama-server could not be reached.
        """
        statuses = await self.model_statuses()
        if model not in statuses:
            raise ModelNotFound(model)
        if statuses[model] == "loaded":
            return
        started = time.monotonic()
        # Step 1: request the load. The router answers at once; the model loads in the background.
        # With one model allowed at a time, llama.cpp refuses a switch while the
        # current model is still busy (for example finishing a generation that
        # was just stopped). That clears by itself, so retry until it does.
        while True:
            try:
                await self._json("POST", "/models/load", json={"model": model})
                break
            except LlamaError as error:
                if "limit reached" not in error.message and error.status != 503:
                    raise
                if time.monotonic() - started > self.load_timeout:
                    raise LoadTimeout(model) from error
                await asyncio.sleep(self.poll_interval)

        # Step 2: poll the status. "loading" then "loaded" is the normal path. "unloaded" after we
        # saw "loading" means the load failed; "unloaded" for longer than start_grace means it never began.
        saw_loading = False
        while True:
            status = (await self.model_statuses()).get(model, "unloaded")
            if status == "loaded":
                return
            if status == "loading":
                saw_loading = True
            elif saw_loading or time.monotonic() - started > self.start_grace:
                raise LoadFailed(model)
            if time.monotonic() - started > self.load_timeout:
                raise LoadTimeout(model)
            await asyncio.sleep(self.poll_interval)


def _error_message(response: httpx.Response) -> str:
    """The message in llama-server's error body, else the raw text.

    llama-server normally sends ``{"error": {"message": ...}}``; this also copes with a
    body that is not JSON or has another shape, so a ``LlamaError`` always carries readable text.
    """
    try:
        error = response.json().get("error", {})
        return error.get("message") or str(error) if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        return response.text or f"HTTP {response.status_code}"
