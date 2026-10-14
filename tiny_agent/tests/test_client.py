"""Tests for the model client and its settings: ``LLM``, ``check``, ``EmbeddingModel`` and ``Settings``.

``node execute.mjs test llm`` runs only this file; ``test agent`` runs everything else.
"""

import json
import os

import httpx
import pytest
from pydantic import ValidationError

from tiny_agent.components.client import check
from tiny_agent.components.client.embedding import EmbeddingModel
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.memory.similarity import cosine
from tiny_agent.components.memory.trimming import TrimmingMemory
from tiny_agent.config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_URL,
    Settings,
    default_env_file,
)
from tiny_agent.main import memory_from_env
from tiny_agent.tests.stubs import EMBEDDING_URL, llm, write_env

# ====================================================================================================
# test_llm
# Model-client contracts across the HTTP boundary.
#
# The caller owns message history while the client formats requests and converts
# server replies. A fake endpoint makes model readiness, request fields and
# failure messages controllable.
#
# What this file demonstrates:
# Tests for tiny_agent.components.client.llm: the LLM client and the Response it returns.
#
# They check the request the client builds (model, messages, tools, reasoning setting),
# how the reply is turned into a ``Response``, the loaded-model check that runs before
# each chat request, and that every failure surfaces as one ``RuntimeError`` with a
# readable message. All against the fake server in ``conftest.py``, never a real model.
#
# Expected observations and limits:
# Recorded bodies show whether tools and reasoning settings were sent; returned
# responses show normalization. Only the first tool call is retained. The tests
# cover selected broken payloads, not every possible malformed optional field.
# ====================================================================================================

MESSAGES = [{"role": "user", "content": "hi"}]


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
    assert isinstance(error.value.__cause__, httpx.TransportError)  # the original is kept


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


# ====================================================================================================
# test_check
# Model status discovery and command exit codes.
#
# Availability differs from readiness: a listed model can be unloaded or still
# loading. Scripted HTTP responses let tests exercise that distinction and the
# separate cases of unknown status, malformed data and an unreachable server.
#
# What this file demonstrates:
# Tests for tiny_agent.components.client.check: which models are loaded.
#
# They cover the three kinds of server (a router that reports ``loaded``/``loading``/
# ``unloaded``, a server that reports no status, and one that cannot be reached or sends
# bad data) and the exit codes of the ``python -m tiny_agent.components.client.check`` command line.
#
# Expected observations and limits:
# The assertions compare returned statuses, printed messages and exit codes.
# A passing fake-server test establishes response handling, not GPU availability
# or compatibility with every model server.
# ====================================================================================================

def test_statuses_of_a_router(fake_server):
    fake_server.router(a="loaded", b="unloaded", c="loading")
    assert check.model_statuses(fake_server.url) == {"a": "loaded", "b": "unloaded", "c": "loading"}


def test_loaded_models_lists_only_loaded(fake_server):
    fake_server.router(a="loaded", b="unloaded")
    assert check.loaded_models(fake_server.url) == ["a"]


def test_loaded_models_empty_when_nothing_is_loaded(fake_server):
    fake_server.router(a="unloaded")
    assert check.loaded_models(fake_server.url) == []


def test_model_status_of_one_model(fake_server):
    fake_server.router(a="loaded", b="loading")
    assert check.model_status("b", fake_server.url) == "loading"


def test_model_status_of_an_unknown_model_is_none(fake_server):
    fake_server.router(a="loaded")
    assert check.model_status("ghost", fake_server.url) is None


def test_a_server_without_status_reports_unknown(fake_server):
    """Ollama and single-model servers list models but say nothing about loading."""
    fake_server.models = [{"id": "gemma4:e4b"}]
    assert check.model_statuses(fake_server.url) == {"gemma4:e4b": "unknown"}
    assert check.loaded_models(fake_server.url) == []


def test_an_empty_server_has_no_models(fake_server):
    assert check.model_statuses(fake_server.url) == {}


def test_unreachable_server_gives_none_not_an_exception(dead_url):
    assert check.model_statuses(dead_url, timeout=1) is None
    assert check.loaded_models(dead_url) == []
    assert check.model_status("a", dead_url) is None


@pytest.mark.parametrize("payload", ["not a list", 5, [1, 2], [{"no_id": True}], [None]])
def test_unexpected_payload_gives_none_not_a_crash(fake_server, payload):
    fake_server.models = payload
    assert check.model_statuses(fake_server.url) is None


def test_trailing_slash_in_the_url_is_fine(fake_server):
    fake_server.router(a="loaded")
    assert check.model_statuses(fake_server.url + "/") == {"a": "loaded"}


# ---- the command line --------------------------------------------------------

def test_cli_lists_models_and_marks_the_loaded_one(fake_server, capsys):
    fake_server.router(a="loaded", b="unloaded")
    assert check.main(["--url", fake_server.url]) == 0
    out = capsys.readouterr().out
    assert "* a" in out and "loaded" in out
    assert "Loaded: a" in out


def test_cli_says_when_nothing_is_loaded(fake_server, capsys):
    fake_server.router(a="unloaded")
    assert check.main(["--url", fake_server.url]) == 0
    assert "none" in capsys.readouterr().out


def test_cli_exit_code_for_a_loaded_model(fake_server, capsys):
    fake_server.router(a="loaded")
    assert check.main(["a", "--url", fake_server.url]) == 0
    assert "a: loaded" in capsys.readouterr().out


def test_cli_exit_code_for_an_unloaded_model(fake_server):
    fake_server.router(a="unloaded")
    assert check.main(["a", "--url", fake_server.url]) == 1


def test_cli_for_an_unknown_model_lists_what_exists(fake_server, capsys):
    fake_server.router(a="loaded", b="unloaded")
    assert check.main(["ghost", "--url", fake_server.url]) == 1
    assert "Available: a, b" in capsys.readouterr().out


def test_cli_exit_code_when_the_server_is_down(dead_url, capsys):
    assert check.main(["--url", dead_url]) == 2
    assert "Cannot reach the model server" in capsys.readouterr().out


def test_cli_treats_unknown_status_as_ok(fake_server):
    fake_server.models = [{"id": "x"}]
    assert check.main(["x", "--url", fake_server.url]) == 0


# ---- command-line edge cases --------------------------------------------------

def test_cli_url_without_a_value_is_a_usage_error_not_a_crash(capsys):
    assert check.main(["--url"]) == 2
    assert "requires an argument" in capsys.readouterr().err


def test_cli_url_followed_by_another_option_is_taken_as_the_url_and_is_unreachable(capsys):
    assert check.main(["--url", "--other"]) == 2
    assert "Cannot reach the model server at --other" in capsys.readouterr().out


def test_cli_url_value_may_come_before_the_model(fake_server):
    fake_server.router(a="loaded")
    assert check.main(["--url", fake_server.url, "a"]) == 0


def test_cli_unknown_option_is_a_usage_error(capsys):
    assert check.main(["--nope"]) == 2
    assert "No such option '--nope'" in capsys.readouterr().err


def test_cli_more_than_one_model_is_a_usage_error(capsys):
    assert check.main(["a", "b"]) == 2
    assert "unexpected extra argument" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_cli_help_prints_usage_and_succeeds(flag, capsys):
    assert check.main([flag]) == 0
    assert "Usage:" in capsys.readouterr().out


def test_cli_does_not_change_the_callers_argument_list(fake_server):
    fake_server.router(a="loaded")
    args = ["a", "--url", fake_server.url]
    check.main(args)
    assert args == ["a", "--url", fake_server.url]


def test_cli_reads_sys_argv_when_called_without_arguments(fake_server, monkeypatch, capsys):
    fake_server.router(a="loaded")
    monkeypatch.setattr("sys.argv", ["check", "a", "--url", fake_server.url])
    assert check.main() == 0
    assert "a: loaded" in capsys.readouterr().out


# ---- model_statuses: a hostile or broken server --------------------------------

@pytest.mark.parametrize("url", ["not a url", "", "ftp//nowhere", "http://", "http://[::1"])
def test_a_malformed_url_gives_none(url):
    assert check.model_statuses(url, timeout=1) is None


def test_a_reply_that_is_not_json_gives_none(fake_server):
    fake_server.raw_models = b"<html>502 Bad Gateway</html>"
    assert check.model_statuses(fake_server.url) is None


def test_a_server_that_hangs_up_gives_none(fake_server):
    fake_server.drop_models = True
    assert check.model_statuses(fake_server.url, timeout=2) is None


def test_a_slow_server_times_out_to_none(fake_server):
    fake_server.models_delay = 2
    assert check.model_statuses(fake_server.url, timeout=0.3) is None


def test_an_http_error_status_gives_none(fake_server):
    fake_server.models_status = 503
    assert check.model_statuses(fake_server.url) is None


# ---- the server address from .env --------------------------------------------------------

def test_the_check_command_takes_its_default_url_from_the_env_file(fake_server, tmp_path, capsys):
    import os

    fake_server.router(a="loaded")
    settings = tmp_path / "settings.env"
    settings.write_text(f"TINYAGENT_LLM_URL={fake_server.url}\n", encoding="utf-8")
    os.environ["TINYAGENT_ENV_FILE"] = str(settings)
    assert check.main([]) == 0  # no --url given
    assert "Loaded: a" in capsys.readouterr().out


def test_the_url_option_beats_the_env_file(fake_server, tmp_path, dead_url, capsys):
    import os

    fake_server.router(a="loaded")
    settings = tmp_path / "settings.env"
    settings.write_text(f"TINYAGENT_LLM_URL={dead_url}\n", encoding="utf-8")
    os.environ["TINYAGENT_ENV_FILE"] = str(settings)
    assert check.main(["--url", fake_server.url]) == 0


# ====================================================================================================
# test_embedding
# Embedding vectors and integration-test evidence.
#
# An embedding maps text into a numeric space, and cosine similarity compares
# vector directions. Real-model tests can expose deployment or output-contract
# problems that a scripted vector cannot.
#
# What this file demonstrates:
# Tests for tiny_agent.components.client.embedding: the ``EmbeddingModel`` client, against the real embedding model.
#
# These talk to EmbeddingGemma as deployed in Docker (the ``llama-embed`` container of ``deploy/llm-client``,
# ``node execute.mjs up llm-client``) at ``TINYAGENT_EMBEDDING_URL`` (default ``http://127.0.0.1:3201/v1``). There
# is no fake: when nothing answers there, the tests that need the model are skipped with the command that starts it.
#
# Expected observations and limits:
# A skip means the server-dependent contract was not verified. The related-text
# comparison uses a few examples; it does not establish embedding quality for
# all domains. The unreachable-server case is independent of a live model.
# ====================================================================================================

def test_embed_returns_a_768_number_vector_of_floats(embedder):
    vector = embedder.embed("Dolphins are amazing!")
    assert len(vector) == 768 and all(isinstance(x, float) for x in vector)


def test_the_same_text_gives_the_same_vector(embedder):
    assert cosine(embedder.embed("I love flamingos."), embedder.embed("I love flamingos.")) == pytest.approx(1.0, abs=1e-4)


def test_related_texts_are_closer_than_unrelated_ones(embedder):
    a = embedder.embed("I love flamingos.")
    b = embedder.embed("Dolphins use echolocation.")
    c = embedder.embed("Flamingos are pink birds.")
    assert cosine(a, c) > cosine(a, b)
    assert cosine(a, c) > cosine(b, c)


def test_a_trailing_slash_on_the_url_is_fine(embedder):
    assert len(EmbeddingModel("embeddinggemma-300m", base_url=EMBEDDING_URL + "/").embed("x")) == 768


def test_an_unreachable_server_becomes_a_runtime_error(dead_url):
    with pytest.raises(RuntimeError, match="Cannot reach the embedding model"):
        EmbeddingModel("embeddinggemma-300m", base_url=dead_url).embed("hi")


# ====================================================================================================
# test_settings
# Typed configuration, precedence and secret-file resolution.
#
# Environment variables are strings, but callers need validated booleans, paths
# and defaults. Temporary dotenv files expose precedence without changing the
# implementation's configuration rules.
#
# What this file demonstrates:
# Tests for ``Settings``: every TINYAGENT_* value, read by pydantic-settings from the environment and the .env file.
#
# Expected observations and limits:
# Assertions cover defaults, environment-over-file priority, flag selection and
# key-file failures. ``Settings()`` still reads the environment; explicit fields
# can override it. A masked representation does not prevent disclosure if a
# caller logs the resolved key.
# ====================================================================================================

def test_defaults_when_nothing_is_set():
    os.environ.pop("TINYAGENT_LOG_DIR", None)
    s = Settings.from_env()
    assert (s.model, s.llm_url, s.think, s.system, s.memory) == (None, DEFAULT_LLM_URL, False, None, "plain")
    assert (s.embedding_model, s.log_dir, s.log_level, s.resolve_api_key()) == (DEFAULT_EMBEDDING_MODEL, None, "INFO", None)


def test_values_come_from_the_environment():
    os.environ.update(TINYAGENT_MODEL="m", TINYAGENT_LLM_URL="http://x/v1", TINYAGENT_THINK="YES",
                      TINYAGENT_SYSTEM="Be brief.", TINYAGENT_LOG_LEVEL="debug", TINYAGENT_LOG_DIR="/logs")
    s = Settings.from_env()
    assert (s.model, s.llm_url, s.think, s.system) == ("m", "http://x/v1", True, "Be brief.")
    assert (s.log_level, str(s.log_dir).replace("\\", "/")) == ("DEBUG", "/logs")


def test_values_come_from_the_env_file(tmp_path):
    write_env(tmp_path, "TINYAGENT_MODEL=from-file\n# a comment\nTINYAGENT_THINK=1\nOTHER_NAME=ignored\n")
    s = Settings.from_env()
    assert (s.model, s.think) == ("from-file", True)
    assert {"model", "think"} <= s.model_fields_set


def test_the_real_environment_wins_over_the_file(tmp_path):
    write_env(tmp_path, "TINYAGENT_MODEL=from-file\nTINYAGENT_SYSTEM=from-file\n")
    os.environ["TINYAGENT_MODEL"] = "from-env"
    s = Settings.from_env()
    assert (s.model, s.system) == ("from-env", "from-file")


def test_an_empty_variable_counts_as_not_set():
    os.environ["TINYAGENT_LLM_URL"] = ""
    os.environ["TINYAGENT_THINK"] = ""
    s = Settings.from_env()
    assert (s.llm_url, s.think) == (DEFAULT_LLM_URL, False)


def test_a_missing_empty_or_folder_env_file_is_not_an_error(tmp_path):
    assert Settings.from_env(tmp_path / "nope.env").model is None
    assert Settings.from_env(write_env(tmp_path, "")).model is None
    assert Settings.from_env(tmp_path).model is None


def test_a_wrong_type_names_the_setting():
    os.environ["TINYAGENT_THINK"] = "maybe"
    with pytest.raises(ValidationError, match="TINYAGENT_THINK|think"):
        Settings.from_env()


def test_the_default_file_is_tiny_agent_dot_env():
    os.environ.pop("TINYAGENT_ENV_FILE")
    assert (default_env_file().name, default_env_file().parent.name) == (".env", "tiny_agent")


def test_tinyagent_env_file_chooses_the_file(tmp_path):
    path = write_env(tmp_path, "")
    assert default_env_file() == path


def test_the_real_dot_env_in_the_repo_loads_without_error():
    os.environ.pop("TINYAGENT_ENV_FILE")
    Settings.from_env()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"TINYAGENT_TRIM_MEMORY": "1"}, "trim"),
        ({"TINYAGENT_TRIM_MEMORY": "1", "TINYAGENT_SUMMARY_MEMORY": "on"}, "summary"),
        ({"TINYAGENT_SUMMARY_MEMORY": "1", "TINYAGENT_RAG_MEMORY": "true"}, "rag"),
        ({"TINYAGENT_RAG_MEMORY": "1", "TINYAGENT_BANK_MEMORY": "1", "TINYAGENT_TRIM_MEMORY": "1"}, "bank"),
        ({"TINYAGENT_BANK_MEMORY": "0"}, "plain"),
    ],
)
def test_the_strongest_memory_flag_wins(env, expected):
    os.environ.update(env)
    assert Settings.from_env().memory == expected


def test_the_api_key_comes_from_a_file_or_the_variable(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("# comment\n\n  secret-from-file  \nsecond\n", encoding="utf-8")
    os.environ["TINYAGENT_API_KEY"] = "from-variable"
    assert Settings.from_env().resolve_api_key() == "from-variable"
    os.environ["TINYAGENT_API_KEY_FILE"] = str(key_file)
    assert Settings.from_env().resolve_api_key() == "secret-from-file"  # the file wins


def test_a_missing_or_empty_key_file_is_an_error(tmp_path):
    os.environ["TINYAGENT_API_KEY_FILE"] = str(tmp_path / "nope")
    with pytest.raises(OSError):
        Settings.from_env().resolve_api_key()
    empty = tmp_path / "empty"
    empty.write_text("# nothing\n", encoding="utf-8")
    os.environ["TINYAGENT_API_KEY_FILE"] = str(empty)
    with pytest.raises(ValueError, match="empty"):
        Settings.from_env().resolve_api_key()


def test_the_key_is_not_shown_in_the_repr():
    os.environ["TINYAGENT_API_KEY"] = "hunter2"
    assert "hunter2" not in repr(Settings.from_env())


def test_settings_can_be_passed_in_code_instead_of_the_environment():
    os.environ["TINYAGENT_TRIM_MEMORY"] = "1"  # ignored: the object given wins
    assert type(memory_from_env(None, Settings())) is TrimmingMemory  # Settings() still reads the environment
    assert type(memory_from_env(None, Settings(trim_memory=False))) is Memory
