"""A stand-in for llama-server in router mode, on an ephemeral port.

It mimics what the real router does (checked against llama.cpp): POST
/models/load answers immediately, the model then reports "loading" and later
"loaded", and loading one model unloads the others (one model at a time).
Special models: "broken" ends up unloaded after loading, "slow" stays loading.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeRouter:
    """A llama-server look-alike a test can inspect and steer.

    Attributes a test can read or set (all put back by :meth:`reset`): ``status`` (model
    to "loaded"/"loading"/"unloaded"), ``chat_calls`` and ``load_calls`` (what the backend
    sent), ``api_key`` (require a bearer token), ``auth_seen`` (every ``Authorization``
    header received), ``health_code``, ``cache_models`` and ``busy_until``.
    """

    LOAD_SECONDS = 0.2  # how long a fake load takes: short, so tests stay fast, but long enough to see "loading"

    def __init__(self):
        """Start the fake server on a free local port in a background thread."""
        self.reset()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def _send(self, code, body, content_type="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                """Like llama-server with --api-key: everything but /health needs the key."""
                fake.auth_seen.append(self.headers.get("Authorization"))
                if fake.api_key is None or self.path == "/health":
                    return True
                if self.headers.get("Authorization") == f"Bearer {fake.api_key}":
                    return True
                self._send(401, {"error": {"code": 401, "message": "Invalid API Key"}})
                return False

            def do_GET(self):
                if not self._authorized():
                    return
                if self.path == "/health":
                    self._send(fake.health_code, {"status": "ok"})
                elif self.path == "/v1/models":
                    entries = [{"id": m, "status": {"value": s}} for m, s in fake.status.items()]
                    if fake.report_source:  # a real router says where each model comes from
                        entries = [{**e, "source": "preset"} for e in entries]
                        entries += [{"id": c, "source": "cache", "status": {"value": "unloaded"}} for c in fake.cache_models]
                    self._send(200, {"data": entries})
                else:
                    self._send(404, {"error": "nope"})

            def do_POST(self):
                if not self._authorized():
                    return
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": {"message": "llama: invalid JSON"}})
                if not isinstance(body, dict):
                    return self._send(400, {"error": {"message": "llama: body must be an object"}})
                if self.path == "/models/load":
                    model = body.get("model")
                    if model not in fake.status:
                        return self._send(404, {"error": {"message": "File Not Found"}})
                    if fake.busy_until > time.monotonic():
                        return self._send(500, {"error": {"code": 500, "message": "model limit reached, try again later"}})
                    fake.load_calls.append(model)
                    fake._begin_load(model)
                    self._send(200, {"success": True})
                elif self.path == "/v1/chat/completions":
                    fake.chat_calls.append(body)
                    if body.get("model") == "explode":
                        return self._send(500, {"error": {"message": "llama crashed"}})
                    if body.get("stream"):
                        chunks = b"".join(
                            b"data: " + json.dumps({"choices": [{"delta": {"content": w}}]}).encode() + b"\n\n"
                            for w in ("Hel", "lo")
                        ) + b"data: [DONE]\n\n"
                        self._send(200, chunks, "text/event-stream")
                    else:
                        self._send(200, {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]})
                else:
                    self._send(404, {"error": "nope"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def reset(self):
        """Back to the starting state: gemma loaded, the others unloaded, no key, no recorded calls."""
        self.status = {"gemma": "loaded", "qwen": "unloaded", "broken": "unloaded", "slow": "unloaded"}
        self.chat_calls = []
        self.load_calls = []
        self.report_source = True  # include a "source" on every model, as llama.cpp's router does
        self.cache_models = []  # extra entries the router found in its download cache
        self.health_code = 200
        self.api_key = None  # when set, requests must carry "Authorization: Bearer <key>"
        self.auth_seen = []  # the Authorization header of every request, in order
        self.busy_until = 0  # monotonic time until which loads are refused as "model limit reached"

    def _begin_load(self, model):
        """Start a fake load: unload the others at once, show "loading", and finish after ``LOAD_SECONDS``."""
        for other in self.status:
            if other != model and self.status[other] == "loaded":
                self.status[other] = "unloaded"
        self.status[model] = "loading"

        def finish():
            time.sleep(self.LOAD_SECONDS)
            if model == "slow":
                return
            self.status[model] = "unloaded" if model == "broken" else "loaded"

        threading.Thread(target=finish, daemon=True).start()

    def close(self):
        """Stop the server and free its port."""
        self.httpd.shutdown()
        self.httpd.server_close()
