"""Backend for the chat app: a FastAPI service in front of llama.cpp's router.

The browser never talks to llama-server directly; it calls this service (through
the frontend's nginx), which calls llama-server over HTTP with httpx.

Environment:
    LLAMA_URL  Base URL of llama-server (default http://llama:8080).

Run:  uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              Backend liveness plus llama-server reachability.
    GET  /api/models          {"models": [{"id", "status"}]}, status loaded|loading|unloaded.
    POST /api/models/load     {"model"}: load it through the router and wait until
                              loaded (200). 404 unknown, 500 load failed, 504 timed
                              out, 502 llama unreachable. One switch at a time.
    POST /api/chat            {"model", "messages"} -> first choice.
    POST /v1/chat/completions Streamed to llama-server, but refused with 409 unless
                              the model is already loaded, so a chat can never
                              trigger a surprise model switch.
    *    /v1/...              Other calls are streamed through unchanged.

Every error body is {"error": {"message", "code"}}.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from llama_client import (
    LlamaClient,
    LlamaError,
    LlamaUnreachable,
    LoadFailed,
    LoadTimeout,
    ModelNotFound,
)
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

# Headers not passed on to llama-server. "authorization" is dropped so a browser can never
# send its own key: the backend always uses the one from its secret file.
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "host", "content-length",
              "authorization"}


def read_secret(path: str | None) -> str | None:
    """Return the first non-empty line of a secret file, or ``None`` if no file is configured.

    Lines starting with ``#`` are comments, as in llama-server's own key file. An
    unreadable or empty file is an error, not a silent "no key": the backend would
    otherwise start and then be refused by llama-server on every request.
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise ValueError(f"The secret file {path} is empty")


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str):
        self.status, self.message, self.code = status, message, code


class LoadRequest(BaseModel):
    model: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[dict]


def create_app(llama_url: str | None = None, **client_options) -> FastAPI:
    """Build the app. ``client_options`` go to LlamaClient (timeouts, polling)."""
    url = llama_url or os.environ.get("LLAMA_URL", "http://llama:8080")
    # The key comes from a file (a Docker secret), never from an environment variable or the image.
    client_options.setdefault("api_key", read_secret(os.environ.get("LLAMA_API_KEY_FILE")))
    # The first load of a model downloads it from Hugging Face (several GB), so allow a long time.
    client_options.setdefault("load_timeout", float(os.environ.get("LLAMA_LOAD_TIMEOUT", "1800")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.llama = LlamaClient(url, **client_options)
        app.state.switch_lock = asyncio.Lock()  # llama.cpp holds one model at a time
        yield
        await app.state.llama.aclose()

    app = FastAPI(title="TinyAgent chat backend", lifespan=lifespan)

    # ---- errors ------------------------------------------------------------

    @app.exception_handler(ApiError)
    async def api_error(_: Request, error: ApiError):
        return JSONResponse({"error": {"message": error.message, "code": error.code}}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, error: RequestValidationError):
        return JSONResponse({"error": {"message": "Invalid request body", "code": "bad_request",
                                       "details": json.loads(json.dumps(error.errors(), default=str))}},
                            status_code=400)

    @app.exception_handler(LlamaUnreachable)
    async def unreachable(_: Request, error: LlamaUnreachable):
        return JSONResponse({"error": {"message": f"Cannot reach llama-server: {error}", "code": "llama_unreachable"}},
                            status_code=502)

    @app.exception_handler(LlamaError)
    async def llama_error(_: Request, error: LlamaError):
        return JSONResponse({"error": {"message": error.message, "code": "llama_error"}}, status_code=error.status)

    @app.exception_handler(404)
    async def not_found(_: Request, __):
        return JSONResponse({"error": {"message": "Not found", "code": "not_found"}}, status_code=404)

    def llama(request: Request) -> LlamaClient:
        return request.app.state.llama

    # ---- endpoints ---------------------------------------------------------

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", "llama": "up" if await llama(request).is_up() else "down"}

    @app.get("/api/models")
    async def models(request: Request):
        statuses = await llama(request).model_statuses()
        return {"models": [{"id": m, "status": s} for m, s in statuses.items()]}

    @app.post("/api/models/load")
    async def load(body: LoadRequest, request: Request):
        try:
            async with request.app.state.switch_lock:
                await llama(request).load_model(body.model)
        except ModelNotFound:
            raise ApiError(404, f"Unknown model: {body.model}", "model_not_found")
        except LoadFailed:
            raise ApiError(500, f"{body.model} failed to load. Check the llama container logs.", "load_failed")
        except LoadTimeout:
            raise ApiError(504, f"{body.model} did not finish loading in time.", "load_timeout")
        return {"model": body.model, "status": "loaded"}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request):
        return await llama(request).chat(body.model, body.messages)

    async def forward(request: Request, path: str, body: bytes | None = None) -> StreamingResponse:
        """Stream a request to llama-server and its reply back, without buffering."""
        client = llama(request)
        if body is None:
            body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        upstream_request = client.http.build_request(
            request.method, path, content=body or None, headers=headers, params=request.query_params)
        try:
            upstream = await client.http.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            raise LlamaUnreachable(str(error) or type(error).__name__) from error
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP},
            background=BackgroundTask(upstream.aclose),
        )

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        raw = await request.body()
        try:
            model = json.loads(raw or b"{}").get("model")
        except (json.JSONDecodeError, AttributeError):
            return await forward(request, "/v1/chat/completions", raw)  # llama-server reports the bad body
        # A model name that is not a string (a list, a number) can never match.
        statuses = await llama(request).model_statuses()
        status = statuses.get(model) if isinstance(model, str) else None
        if status is None:
            raise ApiError(404, f"Unknown model: {model}", "model_not_found")
        if status != "loaded":
            raise ApiError(409, f"{model} is {status}. Load it first (POST /api/models/load).", "model_not_loaded")
        return await forward(request, "/v1/chat/completions", raw)

    @app.get("/v1/models")
    async def v1_models(request: Request):
        """The model list, with the router's cache-discovered duplicates left out (see list_models)."""
        return {"object": "list", "data": await llama(request).list_models()}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def passthrough(path: str, request: Request):
        return await forward(request, f"/v1/{path}")

    return app


app = create_app()
