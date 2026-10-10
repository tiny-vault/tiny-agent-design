"""Backend tests, run against a fake llama.cpp router.  Run:  pytest"""

import json
import threading
import time

from conftest import FAST
from fake_router import FakeRouter
from fastapi.testclient import TestClient
from server import create_app


def statuses(client):
    return {m["id"]: m["status"] for m in client.get("/api/models").json()["models"]}


def completion(client, model, **extra):
    return client.post("/v1/chat/completions",
                       json={"model": model, "messages": [{"role": "user", "content": "hi"}], **extra})


def code(response):
    return response.json()["error"]["code"]


# ---- health and listing -----------------------------------------------------

def test_health_up(client):
    assert client.get("/health").json() == {"status": "ok", "llama": "up"}


def test_health_down(offline_client):
    assert offline_client.get("/health").json() == {"status": "ok", "llama": "down"}


def test_models_lists_statuses(client):
    assert statuses(client) == {"gemma": "loaded", "qwen": "unloaded", "broken": "unloaded", "slow": "unloaded"}


def test_models_502_when_llama_down(offline_client):
    response = offline_client.get("/api/models")
    assert (response.status_code, code(response)) == (502, "llama_unreachable")


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
    assert code(client.get("/nope")) == "not_found"


# ---- model switching --------------------------------------------------------

def test_load_waits_until_loaded_and_router_unloads_the_old_one(client):
    started = time.monotonic()
    response = client.post("/api/models/load", json={"model": "qwen"})
    assert response.status_code == 200
    assert response.json() == {"model": "qwen", "status": "loaded"}
    assert time.monotonic() - started >= FakeRouter.LOAD_SECONDS * 0.8
    assert statuses(client)["qwen"] == "loaded"
    assert statuses(client)["gemma"] == "unloaded"


def test_load_already_loaded_does_not_call_the_router(client, router):
    assert client.post("/api/models/load", json={"model": "gemma"}).status_code == 200
    assert router.load_calls == []


def test_switch_back_and_forth(client):
    for model in ("qwen", "gemma", "qwen"):
        assert client.post("/api/models/load", json={"model": model}).status_code == 200
        assert [m for m, s in statuses(client).items() if s == "loaded"] == [model]


def test_load_unknown_model_404(client):
    response = client.post("/api/models/load", json={"model": "ghost"})
    assert (response.status_code, code(response)) == (404, "model_not_found")


def test_load_failure_500(client):
    response = client.post("/api/models/load", json={"model": "broken"})
    assert (response.status_code, code(response)) == (500, "load_failed")
    assert statuses(client)["broken"] != "loaded"


def test_load_timeout_504(router):
    with TestClient(create_app(router.url, **{**FAST, "load_timeout": 0.4})) as client:
        response = client.post("/api/models/load", json={"model": "slow"})
    assert (response.status_code, code(response)) == (504, "load_timeout")


def test_load_502_when_llama_down(offline_client):
    response = offline_client.post("/api/models/load", json={"model": "qwen"})
    assert (response.status_code, code(response)) == (502, "llama_unreachable")


def test_load_rejects_bad_bodies(client):
    for body in ({}, {"model": ""}, {"model": 5}, {"model": None}):
        response = client.post("/api/models/load", json=body)
        assert (response.status_code, code(response)) == (400, "bad_request"), body
    response = client.post("/api/models/load", content=b"{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400


def test_concurrent_switches_are_serialised(client):
    results = []
    threads = [threading.Thread(target=lambda m=m: results.append(
        client.post("/api/models/load", json={"model": m}).status_code)) for m in ("qwen", "gemma")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200, 200]
    assert len([s for s in statuses(client).values() if s == "loaded"]) == 1


# ---- the chat guard ---------------------------------------------------------

def test_chat_refused_409_when_model_not_loaded_and_never_reaches_llama(client, router):
    response = completion(client, "qwen")
    assert (response.status_code, code(response)) == (409, "model_not_loaded")
    assert router.chat_calls == []
    assert statuses(client)["gemma"] == "loaded"  # no surprise switch


def test_chat_refused_while_loading(client, router):
    router.status["qwen"] = "loading"
    assert completion(client, "qwen").status_code == 409


def test_chat_unknown_model_404(client):
    assert completion(client, "ghost").status_code == 404


def test_chat_allowed_after_load(client, router):
    client.post("/api/models/load", json={"model": "qwen"})
    response = completion(client, "qwen")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello"
    assert len(router.chat_calls) == 1


def test_chat_streams_through(client):
    with client.stream("POST", "/v1/chat/completions",
                       json={"model": "gemma", "stream": True, "messages": []}) as response:
        assert response.status_code == 200
        lines = [l for l in response.iter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert "".join(json.loads(l[5:])["choices"][0]["delta"]["content"] for l in lines[:-1]) == "Hello"


def test_chat_502_when_llama_down(offline_client):
    assert completion(offline_client, "gemma").status_code == 502


def test_other_v1_routes_pass_through(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 4


# ---- /api/chat --------------------------------------------------------------

def test_api_chat_returns_first_choice(client):
    response = client.post("/api/chat", json={"model": "gemma", "messages": [{"role": "user", "content": "hi"}]})
    assert (response.status_code, response.json()["message"]["content"]) == (200, "Hello")


def test_api_chat_requires_model_and_messages(client):
    assert client.post("/api/chat", json={"model": "gemma"}).status_code == 400


# ---- edge cases -------------------------------------------------------------

def test_health_down_when_llama_unhealthy(client, router):
    router.health_code = 503
    assert client.get("/health").json()["llama"] == "down"


def test_chat_with_non_string_or_missing_model_is_unknown(client):
    for model in (None, 5, ["gemma"]):
        response = client.post("/v1/chat/completions", json={"model": model, "messages": []})
        assert response.status_code == 404, model
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 404


def test_chat_with_unicode_model_name_is_unknown(client):
    assert completion(client, "模型-🌍").status_code == 404


def test_chat_with_malformed_json_is_passed_on_not_crashed(client, router):
    response = client.post("/v1/chat/completions", content=b"{oops", headers={"Content-Type": "application/json"})
    assert response.status_code == 400  # llama's own answer
    assert router.chat_calls == []


def test_chat_body_that_is_not_an_object_is_passed_on(client):
    response = client.post("/v1/chat/completions", content=b"[1, 2]", headers={"Content-Type": "application/json"})
    assert response.status_code in (400, 404)


def test_large_chat_body_goes_through_whole(client, router):
    big = "x" * 2_000_000
    response = client.post("/v1/chat/completions", json={"model": "gemma", "messages": [{"role": "user", "content": big}]})
    assert response.status_code == 200
    assert router.chat_calls[0]["messages"][0]["content"] == big


def test_unicode_in_messages_survives(client, router):
    text = "héllo 🌍 日本語"
    client.post("/v1/chat/completions", json={"model": "gemma", "messages": [{"role": "user", "content": text}]})
    assert router.chat_calls[0]["messages"][0]["content"] == text


def test_llama_error_status_is_passed_through_on_v1(client, router):
    router.status["explode"] = "loaded"
    response = client.post("/v1/chat/completions", json={"model": "explode", "messages": []})
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "llama crashed"


def test_api_chat_maps_llama_errors(client):
    response = client.post("/api/chat", json={"model": "explode", "messages": [{"role": "user", "content": "x"}]})
    assert (response.status_code, code(response)) == (500, "llama_error")


def test_api_chat_rejects_wrong_types(client):
    for body in ({"model": "gemma", "messages": "hi"}, {"model": 3, "messages": []}, {"model": "", "messages": []}):
        assert client.post("/api/chat", json=body).status_code == 400, body


def test_wrong_method_is_rejected(client):
    assert client.get("/api/models/load").status_code == 405
    assert client.post("/api/models", json={}).status_code == 405
    assert client.delete("/v1/models").status_code == 405


def test_load_model_name_is_case_sensitive(client):
    assert client.post("/api/models/load", json={"model": "QWEN"}).status_code == 404


def test_load_name_with_odd_characters_is_just_unknown(client):
    for name in ("../etc/passwd", "qwen; rm -rf /", "a" * 5000, "🌍"):
        assert client.post("/api/models/load", json={"model": name}).status_code == 404, name[:20]


def test_load_while_llama_is_mid_switch_waits_for_it(client, router):
    """A second load request must not start before the first one has finished."""
    first = threading.Thread(target=lambda: client.post("/api/models/load", json={"model": "qwen"}))
    first.start()
    time.sleep(0.05)  # qwen is now loading
    started = time.monotonic()
    assert client.post("/api/models/load", json={"model": "gemma"}).status_code == 200
    first.join()
    assert router.load_calls == ["qwen", "gemma"]  # one at a time, in order
    assert time.monotonic() - started >= FakeRouter.LOAD_SECONDS * 0.8


def test_parallel_chats_stream_independently(client):
    results = []

    def chat():
        with client.stream("POST", "/v1/chat/completions", json={"model": "gemma", "stream": True, "messages": []}) as r:
            results.append(sum(1 for line in r.iter_lines() if line.startswith("data:")))

    threads = [threading.Thread(target=chat) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [3] * 5  # two content events and [DONE], for every client


def test_model_list_survives_a_model_without_status(client, router):
    """A single-model llama-server reports no status; its models count as loaded."""
    router.status.clear()
    router.status["only"] = "loaded"
    assert statuses(client) == {"only": "loaded"}
