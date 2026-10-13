"""Tests for tiny_agent.cli: the terminal chat and the ``tiny-agent`` command.

The chat reads lines with ``input`` and prints with ``print``; both are replaceable, so
a test scripts a whole conversation as a list of lines and checks the printed output
and the messages the fake server received. Model choice (``pick_model``), the ``/``
commands, error handling and the exit codes of ``main`` are covered.
"""

import pytest

import cli
from llm_client.llm import LLM


def llm(fake_server, model="a", **kwargs):
    """An LLM pointed at the fake server."""
    return LLM(model=model, base_url=fake_server.url, **kwargs)


def run_chat(client, lines, **kwargs):
    """Run the chat with scripted input; return everything it printed, one entry per output call."""
    feed = iter(lines)

    def input_fn(prompt):
        """Hand out the next scripted line; running out is Ctrl-D (EOFError)."""
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None  # the end of the script is Ctrl-D

    printed = []
    cli.chat(client, input_fn=input_fn, output=printed.append, **kwargs)
    return printed


# ---- pick_model ----------------------------------------------------------------

def test_no_request_uses_the_loaded_model(fake_server):
    fake_server.router(a="unloaded", b="loaded")
    assert cli.pick_model(None, fake_server.url) == ("b", "")


def test_a_requested_loaded_model_is_used(fake_server):
    fake_server.router(a="loaded", b="loaded")
    assert cli.pick_model("b", fake_server.url) == ("b", "")


def test_a_requested_model_that_is_not_offered_falls_back_to_the_loaded_one(fake_server):
    fake_server.router(a="loaded")
    model, note = cli.pick_model("ghost", fake_server.url)
    assert model == "a" and "'ghost' is not offered" in note and "'a'" in note


def test_a_requested_model_that_is_unloaded_is_an_error_with_the_load_command(fake_server):
    fake_server.router(a="loaded", b="unloaded")
    model, note = cli.pick_model("b", fake_server.url)
    assert model is None and "unloaded" in note and "node execute.mjs load b" in note


def test_nothing_loaded_is_an_error(fake_server):
    fake_server.router(a="unloaded")
    model, note = cli.pick_model(None, fake_server.url)
    assert model is None and "No model is loaded" in note


def test_unoffered_and_nothing_loaded_lists_what_exists(fake_server):
    fake_server.router(a="unloaded", b="unloaded")
    model, note = cli.pick_model("ghost", fake_server.url)
    assert model is None and "Available: a, b" in note


def test_a_server_that_is_down_is_an_error(dead_url):
    model, note = cli.pick_model(None, dead_url)
    assert model is None and "Cannot reach the model server" in note


def test_a_server_without_status_takes_the_name_as_given(fake_server):
    fake_server.models = [{"id": "x"}]
    assert cli.pick_model("anything", fake_server.url) == ("anything", "")
    assert cli.pick_model("x", fake_server.url) == ("x", "")


def test_a_server_without_status_and_no_request_starts_with_the_first_listed(fake_server):
    fake_server.models = [{"id": "x"}, {"id": "y"}]
    assert cli.pick_model(None, fake_server.url) == ("x", "")


# ---- the chat loop -----------------------------------------------------------------

def test_the_chat_answers_and_remembers_the_conversation(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("Nice to meet you")
    out = run_chat(llm(fake_server), ["I am Sam", "Who am I?"])
    assert "a> Nice to meet you" in out
    second = fake_server.requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[0]["content"] == "I am Sam" and second[2]["content"] == "Who am I?"


def test_a_system_message_is_sent_first(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    run_chat(llm(fake_server), ["hi"], system="Be brief.")
    assert fake_server.requests[0]["body"]["messages"][0] == {"role": "system", "content": "Be brief."}


def test_blank_lines_are_ignored(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    run_chat(llm(fake_server), ["", "   ", "hi"])
    assert len(fake_server.requests) == 1


def test_exit_and_quit_leave_without_asking_the_model(fake_server):
    fake_server.router(a="loaded")
    for word in ("/exit", "/quit"):
        run_chat(llm(fake_server), [word, "never reached"])
    assert fake_server.requests == []


def test_ctrl_d_and_ctrl_c_leave_cleanly(fake_server):
    fake_server.router(a="loaded")
    assert run_chat(llm(fake_server), [])[-1] == ""  # EOFError

    def interrupt(prompt):
        """Simulate the user pressing Ctrl-C at the prompt."""
        raise KeyboardInterrupt

    printed = []
    cli.chat(llm(fake_server), input_fn=interrupt, output=printed.append)
    assert printed[-1] == ""


def test_clear_forgets_the_conversation_but_keeps_the_system_message(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    run_chat(llm(fake_server), ["first", "/clear", "second"], system="Be brief.")
    after_clear = fake_server.requests[1]["body"]["messages"]
    assert [m["content"] for m in after_clear] == ["Be brief.", "second"]


def test_a_failed_turn_drops_the_question_so_roles_still_alternate(fake_server):
    fake_server.router(a="loaded")
    fake_server.chat_status = 500
    fake_server.chat = {"error": {"message": "boom"}}
    out = run_chat(llm(fake_server), ["first"])
    assert any(line.startswith("error:") and "boom" in line for line in out)

    fake_server.chat_status = 200
    fake_server.reply("fine")
    run_chat(llm(fake_server), ["second"])
    assert [m["role"] for m in fake_server.requests[-1]["body"]["messages"]] == ["user"]


def test_a_reply_with_no_text_is_reported_and_not_remembered(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply(None, tool_calls=[{"id": "1", "function": {"name": "f", "arguments": "{}"}}])
    out = run_chat(llm(fake_server), ["do a thing", "again"])
    assert out.count("(the model sent no text)") == 2
    assert [m["role"] for m in fake_server.requests[1]["body"]["messages"]] == ["user"]


def test_ctrl_c_while_waiting_cancels_the_turn(fake_server):
    class Slow(LLM):
        """An LLM whose request is interrupted by the user (Ctrl-C while waiting)."""

        def generate(self, messages, tools=None):
            """Simulate Ctrl-C arriving while the model is still answering."""
            raise KeyboardInterrupt

    client = Slow(model="a", base_url=fake_server.url)
    out = run_chat(client, ["hello", "/exit"])
    assert any("(cancelled)" in line for line in out)


def test_reasoning_is_shown_only_when_thinking_is_on(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("The answer", reasoning="Let me think")
    quiet = run_chat(llm(fake_server), ["q"])
    loud = run_chat(llm(fake_server, think=True), ["q"])
    assert not any("[thinking]" in line for line in quiet)
    assert "[thinking] Let me think" in loud


# ---- slash commands -----------------------------------------------------------------

def test_help_lists_the_commands(fake_server):
    out = run_chat(llm(fake_server), ["/help"])
    assert any("/models" in line and "/clear" in line for line in out)


def test_models_marks_the_current_one(fake_server):
    fake_server.router(a="loaded", b="unloaded")
    out = run_chat(llm(fake_server), ["/models"])
    assert any(line.startswith("* a") and "loaded" in line for line in out)
    assert any(line.startswith("  b") and "unloaded" in line for line in out)


def test_models_says_when_the_server_is_down(dead_url):
    out = run_chat(LLM(model="a", base_url=dead_url), ["/models"])
    assert any("Cannot reach" in line for line in out)


def test_model_switches_to_another_loaded_model(fake_server):
    fake_server.router(a="loaded", b="loaded")
    fake_server.reply("ok")
    client = llm(fake_server)
    out = run_chat(client, ["/model b", "hi"])
    assert "Now using b." in out
    assert client.model == "b"
    assert fake_server.requests[0]["body"]["model"] == "b"


def test_model_refuses_an_unloaded_one_and_keeps_the_current(fake_server):
    fake_server.router(a="loaded", b="unloaded")
    client = llm(fake_server)
    out = run_chat(client, ["/model b"])
    assert any("unloaded" in line for line in out)
    assert client.model == "a"


def test_model_without_a_name_shows_the_current_one(fake_server):
    assert "Current model: a" in run_chat(llm(fake_server), ["/model"])


def test_think_toggles_and_reports(fake_server):
    client = llm(fake_server)
    out = run_chat(client, ["/think", "/think on", "/think off", "/think maybe"])
    assert "Thinking is off. Use /think on or /think off." in out[1]
    assert "Thinking is on." in out and "Thinking is off." in out
    assert client.think is False


def test_an_unknown_command_is_reported_not_sent_to_the_model(fake_server):
    fake_server.router(a="loaded")
    out = run_chat(llm(fake_server), ["/frobnicate"])
    assert any("Unknown command: /frobnicate" in line for line in out)
    assert fake_server.requests == []


# ---- the command ---------------------------------------------------------------------

def test_one_shot_prints_the_answer_and_exits_zero(fake_server, capsys):
    fake_server.router(a="loaded")
    fake_server.reply("42")
    assert cli.main(["--url", fake_server.url, "What", "is", "6x7?"]) == 0
    assert capsys.readouterr().out.strip() == "42"
    assert fake_server.requests[0]["body"]["messages"] == [{"role": "user", "content": "What is 6x7?"}]


def test_one_shot_with_a_system_message(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    cli.main(["--url", fake_server.url, "--system", "Be brief.", "hi"])
    assert fake_server.requests[0]["body"]["messages"][0] == {"role": "system", "content": "Be brief."}


def test_one_shot_error_exits_one(fake_server, capsys):
    fake_server.router(a="loaded")
    fake_server.chat_status = 500
    fake_server.chat = {"error": {"message": "boom"}}
    assert cli.main(["--url", fake_server.url, "hi"]) == 1
    assert "error:" in capsys.readouterr().out


def test_one_shot_think_flag_reaches_the_model(fake_server):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    cli.main(["--url", fake_server.url, "--think", "hi"])
    assert "reasoning_effort" not in fake_server.requests[0]["body"]


def test_exit_code_two_when_the_server_is_down(dead_url, capsys):
    assert cli.main(["--url", dead_url, "hi"]) == 2
    assert "Cannot reach the model server" in capsys.readouterr().out


def test_exit_code_two_when_nothing_is_loaded(fake_server, capsys):
    fake_server.router(a="unloaded")
    assert cli.main(["--url", fake_server.url, "hi"]) == 2
    assert "No model is loaded" in capsys.readouterr().out


def test_the_model_option_is_used(fake_server):
    fake_server.router(a="loaded", b="loaded")
    fake_server.reply("ok")
    cli.main(["--url", fake_server.url, "-m", "b", "hi"])
    assert fake_server.requests[0]["body"]["model"] == "b"


def test_an_unoffered_model_option_says_what_was_used_instead(fake_server, capsys):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    assert cli.main(["--url", fake_server.url, "-m", "ghost", "hi"]) == 0
    assert "using the loaded model 'a'" in capsys.readouterr().out
    assert fake_server.requests[0]["body"]["model"] == "a"


def test_the_environment_sets_the_defaults(fake_server, monkeypatch):
    fake_server.router(a="loaded", b="loaded")
    fake_server.reply("ok")
    monkeypatch.setenv("TINYAGENT_LLM_URL", fake_server.url)
    monkeypatch.setenv("TINYAGENT_MODEL", "b")
    assert cli.main(["hi"]) == 0
    assert fake_server.requests[0]["body"]["model"] == "b"


def test_interactive_mode_runs_when_there_is_no_message(fake_server, monkeypatch, capsys):
    fake_server.router(a="loaded")
    fake_server.reply("hello there")
    lines = iter(["hi", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["--url", fake_server.url]) == 0
    assert "a> hello there" in capsys.readouterr().out


def test_demo_runs_main_first_and_a_demo_failure_does_not_stop_the_chat(fake_server, monkeypatch, capsys):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    import main as demo

    def broken_demo():
        """A demo whose model call fails."""
        raise RuntimeError("demo model is missing")

    monkeypatch.setattr(demo, "main", broken_demo)
    assert cli.main(["--url", fake_server.url, "--demo", "hi"]) == 0
    out = capsys.readouterr().out
    assert "demo failed: demo model is missing" in out and "ok" in out


def test_bad_options_exit_with_a_usage_error():
    with pytest.raises(SystemExit) as error:
        cli.main(["--no-such-option"])
    assert error.value.code == 2


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "tiny-agent" in capsys.readouterr().out


# ---- settings from code/.env ---------------------------------------------------------------

def env_file(tmp_path, text):
    """Write a settings file and make the code read it instead of the real code/.env."""
    import os

    path = tmp_path / "settings.env"
    path.write_text(text, encoding="utf-8")
    os.environ["TINYAGENT_ENV_FILE"] = str(path)
    return path


def test_the_env_file_supplies_the_url_and_the_model(fake_server, tmp_path):
    fake_server.router(a="loaded", b="loaded")
    fake_server.reply("ok")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_MODEL=b\n")
    assert cli.main(["hi"]) == 0  # no --url, no --model on the command line
    assert fake_server.requests[0]["body"]["model"] == "b"


def test_command_line_options_beat_the_env_file(fake_server, tmp_path):
    fake_server.router(a="loaded", b="loaded")
    fake_server.reply("ok")
    env_file(tmp_path, f"TINYAGENT_LLM_URL=http://127.0.0.1:9/v1\nTINYAGENT_MODEL=a\n")
    assert cli.main(["--url", fake_server.url, "-m", "b", "hi"]) == 0
    assert fake_server.requests[0]["body"]["model"] == "b"


def test_the_real_environment_beats_the_env_file(fake_server, tmp_path, monkeypatch):
    fake_server.router(a="loaded", b="loaded")
    fake_server.reply("ok")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_MODEL=a\n")
    monkeypatch.setenv("TINYAGENT_MODEL", "b")
    assert cli.main(["hi"]) == 0
    assert fake_server.requests[0]["body"]["model"] == "b"


def test_think_and_system_come_from_the_env_file(fake_server, tmp_path):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    backslash_n = chr(92) + "n"  # the two characters \ and n, which the file format turns into a new line
    env_file(tmp_path, f'TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_THINK=yes\nTINYAGENT_SYSTEM="Be brief.{backslash_n}Really."\n')
    cli.main(["hi"])
    body = fake_server.requests[0]["body"]
    assert "reasoning_effort" not in body  # thinking is on
    assert body["messages"][0] == {"role": "system", "content": "Be brief.\nReally."}


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("on", True), (" On ", True),
                                            ("0", False), ("false", False), ("no", False), ("", False), (None, False), ("maybe", False)])
def test_truthy(value, expected):
    assert cli.truthy(value) is expected


def test_an_api_key_from_the_env_file_is_sent(fake_server, tmp_path):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_API_KEY=sekret-123\n")
    cli.main(["hi"])
    assert fake_server.requests[0]["headers"]["Authorization"] == "Bearer sekret-123"


def test_an_api_key_file_is_read_and_beats_nothing_else_is_needed(fake_server, tmp_path):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    key = tmp_path / "api_key"
    key.write_text("# the key\n\nfile-key-456\n", encoding="utf-8")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_API_KEY_FILE={key}\nTINYAGENT_API_KEY=ignored\n")
    cli.main(["hi"])
    assert fake_server.requests[0]["headers"]["Authorization"] == "Bearer file-key-456"


def test_no_key_configured_keeps_the_placeholder(fake_server, tmp_path):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\n")
    cli.main(["hi"])
    assert fake_server.requests[0]["headers"]["Authorization"] == "Bearer no_key"


def test_a_missing_key_file_is_a_clear_error_not_a_crash(fake_server, tmp_path, capsys):
    fake_server.router(a="loaded")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_API_KEY_FILE={tmp_path / 'nope'}\n")
    assert cli.main(["hi"]) == 2
    assert "Cannot read the API key file" in capsys.readouterr().out
    assert fake_server.requests == []


def test_an_empty_key_file_is_a_clear_error(fake_server, tmp_path, capsys):
    fake_server.router(a="loaded")
    key = tmp_path / "api_key"
    key.write_text("# nothing\n", encoding="utf-8")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_API_KEY_FILE={key}\n")
    assert cli.main(["hi"]) == 2
    assert "is empty" in capsys.readouterr().out


def test_a_missing_or_empty_env_file_changes_nothing(fake_server, tmp_path):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    (tmp_path / "empty.env").write_text("", encoding="utf-8")
    import os
    for target in (tmp_path / "empty.env", tmp_path / "missing.env"):
        os.environ["TINYAGENT_ENV_FILE"] = str(target)
        assert cli.main(["--url", fake_server.url, "hi"]) == 0


def test_the_log_names_the_loaded_settings_but_never_their_values(fake_server, tmp_path, log_folder):
    from tiny_agent import log

    fake_server.router(a="loaded")
    fake_server.reply("ok")
    env_file(tmp_path, f"TINYAGENT_LLM_URL={fake_server.url}\nTINYAGENT_API_KEY=very-secret-value\n")
    cli.main(["hi"])
    text = (log_folder / "cli.log").read_text(encoding="utf-8")
    assert "settings loaded from the .env file: TINYAGENT_API_KEY, TINYAGENT_LLM_URL" in text
    assert "very-secret-value" not in text
    for handler in list(__import__("logging").getLogger("tiny_agent.cli").handlers):
        if getattr(handler, "_tiny_agent_file_handler", False):
            handler.close()


def test_main_py_reads_its_log_settings_from_the_env_file(tmp_path, demo_server, capsys):
    import logging

    import main as demo

    import os

    os.environ.pop("TINYAGENT_LOG_DIR")  # the log_folder fixture sets it; the real environment would win over the file
    logs = tmp_path / "from-env-file"
    env_file(tmp_path, f"TINYAGENT_LOG_DIR={logs}\n")
    demo.main()
    assert (logs / "main.log").exists()
    for handler in list(logging.getLogger("tiny_agent.main").handlers):
        if getattr(handler, "_tiny_agent_file_handler", False):
            handler.close()
