"""Shared fixtures: a fake OpenAI-compatible model server on an ephemeral port.

Set ``fake_server.models`` to what ``GET /v1/models`` should return and
``fake_server.chat`` to what ``POST /v1/chat/completions`` should return. Every
chat request body is recorded in ``fake_server.requests``.

Why a fake server: the code under test talks HTTP to a model server. A real model
is slow, needs a GPU and answers differently each time, so the tests start a tiny
server on the local machine that returns exactly what the test sets. The tests then
check what the code sent and how it handled the reply, deterministically and offline.
The autouse fixtures also isolate each test from the real ``.env`` file and log folder.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# Make `tiny_agent` importable no matter where pytest is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeServer:
    """A stand-in for a model server, serving two endpoints on 127.0.0.1.

    * ``GET  <url>/models``: returns ``{"data": self.models}``.
    * ``POST <url>/chat/completions``: records the request in ``self.requests`` and
      returns ``self.chat`` with status ``self.chat_status``.

    Attributes:
        url: The API root to give to ``LLM(base_url=...)`` or ``check``, ending in
            ``/v1``. The port is chosen by the OS so tests never collide.
        models: What ``/models`` returns under ``data``.
        chat: The JSON body ``/chat/completions`` returns.
        chat_status: The HTTP status ``/chat/completions`` returns.
        requests: One ``{"body", "headers"}`` dict per chat request received, in
            order. Empty means the model was never called.
        models_status: HTTP status ``/models`` answers with (200 by default).
        raw_models, raw_chat: Bytes to send instead of JSON, to simulate a server
            that answers with something that is not JSON (``None`` = normal).
        drop_models, drop_chat: Hang up without answering, to simulate a crashed
            or reset connection.
        models_delay: Seconds ``/models`` waits before answering, to simulate a
            slow server.
    """

    def __init__(self):
        """Start the server on a free port in a background thread."""
        self.models = []
        self.chat = {}
        self.chat_status = 200
        self.requests = []
        # Ways to misbehave, for the robustness tests. All off by default.
        self.models_status = 200  # HTTP status of /models
        self.raw_models = None  # bytes to send from /models instead of JSON
        self.raw_chat = None  # bytes to send from /chat/completions instead of JSON
        self.drop_models = False  # hang up on /models without answering
        self.drop_chat = False  # hang up on /chat/completions without answering
        self.models_delay = 0  # seconds /models waits before answering
        fake = self

        class Handler(BaseHTTPRequestHandler):
            """Answers requests from the FakeServer's current state."""

            def log_message(self, format, *args):
                """Stay quiet: do not print a line per request."""

            def _send(self, code, body):
                """Reply with ``body`` as JSON and HTTP status ``code``."""
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                """Serve the model list for any GET (only ``/models`` is used)."""
                if fake.models_delay:
                    time.sleep(fake.models_delay)
                if fake.drop_models:
                    self.close_connection = True  # hang up without a reply
                    return
                self._send(fake.models_status, fake.raw_models if fake.raw_models is not None else {"data": fake.models})

            def do_POST(self):
                """Record the chat request, then reply with the scripted answer."""
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                fake.requests.append({"body": body, "headers": dict(self.headers)})
                if fake.drop_chat:
                    self.close_connection = True  # hang up without a reply
                    return
                self._send(fake.chat_status, fake.raw_chat if fake.raw_chat is not None else fake.chat)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def router(self, **statuses):
        """Make the server look like a llama.cpp router: ``router(a="loaded", b="unloaded")``."""
        self.models = [{"id": name, "status": {"value": status}} for name, status in statuses.items()]

    def reply(self, content="ok", **message):
        """Set a normal chat reply, with usage.

        Args:
            content: The answer text (``None`` for a tool-call-only reply).
            **message: Extra fields on the reply message, for example
                ``reasoning="..."`` or ``tool_calls=[...]``.
        """
        self.chat = {
            "model": "m",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            "choices": [{"message": {"content": content, **message}}],
        }

    def close(self):
        """Stop the server and free its port."""
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_server():
    """A fresh FakeServer for one test, stopped afterwards."""
    server = FakeServer()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep every test away from the real ``code/.env`` and stop environment changes leaking.

    ``os.environ`` is swapped for a copy (put back after the test), the settings file
    is pointed at a file that does not exist, and the TINYAGENT_* settings are cleared.
    A test that wants a settings file writes one and sets TINYAGENT_ENV_FILE itself.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ["TINYAGENT_ENV_FILE"] = str(tmp_path / "no-such.env")
    for name in ("TINYAGENT_MODEL", "TINYAGENT_LLM_URL", "TINYAGENT_API_KEY", "TINYAGENT_API_KEY_FILE",
                 "TINYAGENT_THINK", "TINYAGENT_SYSTEM"):
        os.environ.pop(name, None)


@pytest.fixture
def demo_server(fake_server, monkeypatch):
    """A FakeServer that ``main.py`` talks to: ``gemma4-e4b`` loaded, answering ``Your names are Maarten and Jay.``."""
    fake_server.router(**{"gemma4-e4b": "loaded"})
    fake_server.reply("Your names are Maarten and Jay.")
    monkeypatch.setenv("TINYAGENT_LLM_URL", fake_server.url)
    return fake_server


@pytest.fixture
def dead_url():
    """A model server URL where nothing is listening."""
    return "http://127.0.0.1:9/v1"


@pytest.fixture(autouse=True)
def log_folder(tmp_path, monkeypatch):
    """Send every test's log files to a temp folder, never the repo's ``logs/``.

    Returns the folder, so a test can read what was logged.
    """
    folder = tmp_path / "logs"
    monkeypatch.setenv("TINYAGENT_LOG_DIR", str(folder))
    return folder
