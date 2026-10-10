"""Tests for llm_client.check: which models are loaded."""

import pytest

from llm_client import check


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
    assert "--url needs a value" in capsys.readouterr().out


def test_cli_url_followed_by_another_option_is_a_usage_error(capsys):
    assert check.main(["--url", "--other"]) == 2
    assert "--url needs a value" in capsys.readouterr().out


def test_cli_url_value_may_come_before_the_model(fake_server):
    fake_server.router(a="loaded")
    assert check.main(["--url", fake_server.url, "a"]) == 0


def test_cli_unknown_option_is_a_usage_error(capsys):
    assert check.main(["--nope"]) == 2
    assert "Unknown option: --nope" in capsys.readouterr().out


def test_cli_more_than_one_model_is_a_usage_error(capsys):
    assert check.main(["a", "b"]) == 2
    assert "at most one model" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_cli_help_prints_usage_and_succeeds(flag, capsys):
    assert check.main([flag]) == 0
    assert "usage:" in capsys.readouterr().out


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


# ---- the server address from code/.env --------------------------------------------------------

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
