"""Async client for llama-server's HTTP API (router mode), built on httpx.

The request shape for chat follows the example in *An Illustrated Guide to AI
Agents* by Maarten Grootendorst and Jay Alammar (O'Reilly): POST ``model`` and
``messages`` to ``/v1/chat/completions`` and read ``choices[0]``.
"""

import asyncio
import time

import httpx


class LlamaUnreachable(Exception):
    """llama-server could not be reached."""


class LlamaError(Exception):
    """llama-server answered with an error status."""

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
        self.base_url = base_url.rstrip("/")
        # llama-server may require a key (--api-key-file). It is sent as a bearer token on every request.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                      timeout=httpx.Timeout(timeout, connect=5))
        self.load_timeout = load_timeout
        self.poll_interval = poll_interval
        self.start_grace = start_grace

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _json(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = await self.http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise LlamaUnreachable(str(error) or type(error).__name__) from error
        if response.is_error:
            raise LlamaError(response.status_code, _error_message(response))
        return response.json()

    async def is_up(self) -> bool:
        try:
            return (await self.http.get("/health", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def model_statuses(self) -> dict[str, str]:
        """Return ``{model: status}`` with status loaded, loading or unloaded.

        A server without router mode reports no status; its models count as loaded.
        """
        data = (await self._json("GET", "/v1/models")).get("data", [])
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
    """The message in llama-server's error body, else the raw text."""
    try:
        error = response.json().get("error", {})
        return error.get("message") or str(error) if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        return response.text or f"HTTP {response.status_code}"
