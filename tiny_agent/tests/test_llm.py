"""Tests for tiny_agent.llm_client.llm: the LLM client and the Response it returns.

They check the request the client builds (model, messages, tools, reasoning setting),
how the reply is turned into a ``Response``, the loaded-model check that runs before
each chat request, and that every failure surfaces as one ``RuntimeError`` with a
readable message. All against the fake server in ``conftest.py``, never a real model.
"""

import json

import pytest

from tiny_agent.llm_client.llm import LLM
from tiny_agent.llm_client.response import Response

MESSAGES = [{"role": "user", "content": "hi"}]


def llm(fake_server, model="a", **kwargs):
    """Build an LLM pointed at the fake server; ``kwargs`` go to ``LLM(...)``."""
    return LLM(model=model, base_url=fake_server.url, **kwargs)


# ---- a normal reply ------------------------------------------------------------

def test_generate_returns_a_response_with_the_content(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("pong")
    response = llm(fake_server).generate(MESSAGES)
    assert isinstance(response, Response)
    assert response.content == "pong"
    assert response.tool_call is None


def test_generate_reports_token_usage_in_metadata(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("x")
    assert llm(fake_server).generate(MESSAGES).metadata == {"model": "m", "prompt_tokens": 3, "completion_tokens": 4}


def test_missing_usage_and_model_do_not_crash(fake_server):
    fake_server.router(a="loaded")
    fake_server.chat = {"choices": [{"message": {"content": "pong"}}]}
    response = llm(fake_server).generate(MESSAGES)
    assert response.metadata == {"model": "a", "prompt_tokens": None, "completion_tokens": None}


def test_null_content_stays_none(fake_server):
    fake_server.router(a="loaded")
    fake_server.chat = {"model": "a", "usage": {}, "choices": [{"message": {"content": None}}]}
    assert llm(fake_server).generate(MESSAGES).content is None


# ---- what is sent ----------------------------------------------------------------

def test_request_carries_model_messages_and_auth(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply()
    llm(fake_server, api_key="secret").generate(MESSAGES)
    sent = fake_server.requests[0]
    assert sent["body"]["model"] == "a"
    assert sent["body"]["messages"] == MESSAGES
    assert sent["headers"]["Authorization"] == "Bearer secret"


def test_tools_are_sent_only_when_given(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply()
    tools = [{"type": "function", "function": {"name": "f"}}]
    llm(fake_server).generate(MESSAGES)
    llm(fake_server).generate(MESSAGES, tools=tools)
    assert "tools" not in fake_server.requests[0]["body"]
    assert fake_server.requests[1]["body"]["tools"] == tools


def test_thinking_is_switched_off_unless_asked_for(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply()
    llm(fake_server).generate(MESSAGES)
    llm(fake_server, think=True).generate(MESSAGES)
    assert fake_server.requests[0]["body"]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in fake_server.requests[1]["body"]


def test_default_url_is_the_docker_gateway():
    assert LLM(model="a").base_url == "http://127.0.0.1:3200/v1"


# ---- reasoning and tool calls --------------------------------------------------------

@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_reasoning_is_read_under_either_name(fake_server, key):
    """Ollama says "reasoning", llama.cpp says "reasoning_content"."""
    fake_server.router(a="loaded")
    fake_server.reply("x", **{key: "thinking"})
    assert llm(fake_server).generate(MESSAGES).reasoning == "thinking"


def test_no_reasoning_is_none(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("x")
    assert llm(fake_server).generate(MESSAGES).reasoning is None


def test_a_tool_call_is_returned(fake_server):
    fake_server.router(a="loaded")
    call = {"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
    fake_server.reply(None, tool_calls=[call])
    response = llm(fake_server).generate(MESSAGES)
    assert response.tool_call == call
    assert response.tool_call is not None
    assert json.loads(response.tool_call["function"]["arguments"]) == {"city": "Paris"}


def test_only_the_first_of_several_tool_calls_is_returned(fake_server):
    """Known limit of Response.tool_call being a single dict."""
    fake_server.router(a="loaded")
    calls = [{"id": "1", "function": {"name": "f", "arguments": "{}"}}, {"id": "2", "function": {"name": "g", "arguments": "{}"}}]
    fake_server.reply(None, tool_calls=calls)
    assert llm(fake_server).generate(MESSAGES).tool_call["id"] == "1"  # pyright: ignore[reportOptionalSubscript]


def test_empty_tool_calls_list_means_no_tool_call(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("answer", tool_calls=[])
    assert llm(fake_server).generate(MESSAGES).tool_call is None


# ---- the loaded-model check ---------------------------------------------------------------

@pytest.mark.parametrize("model,word", [("b", "unloaded"), ("c", "loading")])
def test_a_model_that_is_not_loaded_is_refused_before_any_request(fake_server, model, word):
    fake_server.router(a="loaded", b="unloaded", c="loading")
    fake_server.reply()
    with pytest.raises(RuntimeError) as error:
        llm(fake_server, model=model).generate(MESSAGES)
    assert word in str(error.value)
    assert "loaded now: a" in str(error.value)
    assert f"node execute.mjs load {model}" in str(error.value)
    assert fake_server.requests == []  # nothing was sent to the model


def test_the_message_says_none_when_nothing_is_loaded(fake_server):
    fake_server.router(a="unloaded")
    with pytest.raises(RuntimeError, match="loaded now: none"):
        llm(fake_server).generate(MESSAGES)


def test_a_server_without_status_is_not_blocked(fake_server):
    """Ollama lists models but does not say which are loaded, so do not guess."""
    fake_server.models = [{"id": "a"}]
    fake_server.reply("ok")
    assert llm(fake_server).generate(MESSAGES).content == "ok"


def test_a_model_the_router_does_not_offer_uses_the_loaded_one(fake_server):
    fake_server.router(other="loaded", spare="unloaded")
    fake_server.reply("ok")
    with pytest.warns(UserWarning, match=r"'a' is not offered.*using the loaded model 'other'"):
        response = llm(fake_server).generate(MESSAGES)
    assert response.content == "ok"
    assert fake_server.requests[0]["body"]["model"] == "other"  # the loaded model was asked, not "a"


def test_the_substitution_is_for_one_call_only(fake_server):
    fake_server.router(other="loaded")
    fake_server.reply("ok")
    client = llm(fake_server)
    with pytest.warns(UserWarning):
        client.generate(MESSAGES)
    assert client.model == "a"  # the configured model is not rewritten


def test_the_first_loaded_model_is_chosen_when_several_are(fake_server):
    fake_server.router(x="unloaded", y="loaded", z="loaded")
    fake_server.reply("ok")
    with pytest.warns(UserWarning):
        llm(fake_server).generate(MESSAGES)
    assert fake_server.requests[0]["body"]["model"] == "y"


def test_a_loading_model_is_not_picked_as_a_substitute(fake_server):
    fake_server.router(x="loading", y="loaded")
    fake_server.reply("ok")
    with pytest.warns(UserWarning):
        llm(fake_server).generate(MESSAGES)
    assert fake_server.requests[0]["body"]["model"] == "y"


def test_an_unoffered_model_with_nothing_loaded_lists_what_is_available(fake_server):
    fake_server.router(x="unloaded", y="unloaded")
    with pytest.raises(RuntimeError, match=r"not offered and no model is loaded.*Available: x, y"):
        llm(fake_server).generate(MESSAGES)
    assert fake_server.requests == []


def test_a_model_that_is_offered_but_unloaded_is_not_substituted(fake_server):
    """Substitution is only for names the router does not offer; an offered, unloaded model is an error."""
    fake_server.router(a="unloaded", other="loaded")
    with pytest.raises(RuntimeError, match="is unloaded"):
        llm(fake_server).generate(MESSAGES)
    assert fake_server.requests == []


def test_a_loaded_model_is_used_without_a_warning(fake_server, recwarn):
    fake_server.router(a="loaded", other="loaded")
    fake_server.reply("ok")
    llm(fake_server).generate(MESSAGES)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    assert fake_server.requests[0]["body"]["model"] == "a"


def test_a_server_without_status_never_substitutes(fake_server, recwarn):
    """Ollama lists installed models but not what is loaded, so the name is sent as given."""
    fake_server.models = [{"id": "other"}]
    fake_server.reply("ok")
    assert llm(fake_server).generate(MESSAGES).content == "ok"
    assert fake_server.requests[0]["body"]["model"] == "a"
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_metadata_reports_the_model_that_really_answered(fake_server):
    fake_server.router(other="loaded")
    fake_server.chat = {"model": "other", "usage": {}, "choices": [{"message": {"content": "ok"}}]}
    with pytest.warns(UserWarning):
        assert llm(fake_server).generate(MESSAGES).metadata["model"] == "other"  # pyright: ignore[reportOptionalSubscript]


# ---- failures: every one becomes a RuntimeError that says what happened ----------------

def test_unreachable_server_raises_a_clear_error(dead_url):
    with pytest.raises(RuntimeError, match="Cannot reach the model") as error:
        LLM(model="a", base_url=dead_url).generate(MESSAGES)
    assert isinstance(error.value.__cause__, OSError)  # the original is kept


def test_an_http_error_carries_the_servers_message(fake_server):
    fake_server.router(a="loaded")
    fake_server.chat_status = 500
    fake_server.chat = {"error": {"message": "out of memory"}}
    with pytest.raises(RuntimeError, match=r"HTTP 500 .*out of memory"):
        llm(fake_server).generate(MESSAGES)


def test_an_http_error_without_a_json_body_still_reports_the_status(fake_server):
    fake_server.router(a="loaded")
    fake_server.chat_status = 404
    fake_server.chat = "plain text, not an error object"
    with pytest.raises(RuntimeError, match="HTTP 404"):
        llm(fake_server).generate(MESSAGES)


@pytest.mark.parametrize("reply", [{"nothing": "here"}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": None}]},
                                   {"choices": "oops"}, [1, 2, 3]])
def test_a_reply_that_is_not_a_chat_completion_raises(fake_server, reply):
    fake_server.router(a="loaded")
    fake_server.chat = reply
    with pytest.raises(RuntimeError, match="Unexpected reply"):
        llm(fake_server).generate(MESSAGES)


def test_a_reply_that_is_not_json_raises(fake_server):
    fake_server.router(a="loaded")
    fake_server.raw_chat = b"<html>502 Bad Gateway</html>"
    with pytest.raises(RuntimeError, match="not JSON"):
        llm(fake_server).generate(MESSAGES)


@pytest.mark.parametrize("url", ["not a url", "", "ftp//nowhere"])
def test_a_malformed_url_raises_not_crashes(url):
    with pytest.raises(RuntimeError):
        LLM(model="a", base_url=url).generate(MESSAGES)


def test_empty_messages_are_refused_before_anything_is_sent(fake_server):
    fake_server.router(a="loaded")
    with pytest.raises(ValueError, match="must not be empty"):
        llm(fake_server).generate([])
    assert fake_server.requests == []


def test_a_trailing_slash_in_base_url_is_fine(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    assert LLM(model="a", base_url=fake_server.url + "/").generate(MESSAGES).content == "ok"


def test_a_dropped_connection_mid_request_raises_a_clear_error(fake_server):
    fake_server.router(a="loaded")
    fake_server.drop_chat = True  # the server hangs up without answering
    with pytest.raises(RuntimeError, match="Cannot reach the model"):
        llm(fake_server).generate(MESSAGES)


def test_unicode_in_messages_and_replies_survives(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("héllo 🌍 日本語")
    response = llm(fake_server).generate([{"role": "user", "content": "grüß dich 🌍"}])
    assert response.content == "héllo 🌍 日本語"
    assert fake_server.requests[0]["body"]["messages"][0]["content"] == "grüß dich 🌍"
