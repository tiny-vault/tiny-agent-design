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
    LOAD_SECONDS = 0.2

    def __init__(self):
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

            def do_GET(self):
                if self.path == "/health":
                    self._send(fake.health_code, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": m, "status": {"value": s}} for m, s in fake.status.items()]})
                else:
                    self._send(404, {"error": "nope"})

            def do_POST(self):
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
        self.status = {"gemma": "loaded", "qwen": "unloaded", "broken": "unloaded", "slow": "unloaded"}
        self.chat_calls = []
        self.load_calls = []
        self.health_code = 200
        self.busy_until = 0  # monotonic time until which loads are refused as "model limit reached"

    def _begin_load(self, model):
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
        self.httpd.shutdown()
        self.httpd.server_close()
