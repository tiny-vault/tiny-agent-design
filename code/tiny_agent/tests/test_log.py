"""Tests for tiny_agent.log (the log files) and for what cli.py and main.py write to them."""

import logging
from pathlib import Path

import pytest

import cli
import main as demo
from tiny_agent import log


@pytest.fixture(autouse=True)
def clean_loggers():
    """Close and detach the file handlers each test created, so files can be deleted."""
    yield
    for name in ("cli", "main", "probe"):
        logger = logging.getLogger(f"tiny_agent.{name}")
        for handler in list(logger.handlers):
            if getattr(handler, "_tiny_agent_file_handler", False):
                logger.removeHandler(handler)
                handler.close()


def feed(lines):
    """An input function that hands out the scripted lines, one per prompt."""
    remaining = iter(lines)
    return lambda prompt="": next(remaining)


def read(folder, name):
    """The text of ``<folder>/<name>.log``."""
    return (folder / f"{name}.log").read_text(encoding="utf-8")


# ---- tiny_agent.log -----------------------------------------------------------------

def test_log_dir_comes_from_the_environment(log_folder):
    assert log.log_dir() == log_folder


def test_log_dir_defaults_to_experimentation_logs_in_the_repo(monkeypatch):
    monkeypatch.delenv("TINYAGENT_LOG_DIR")
    expected = Path(log.__file__).resolve().parents[2] / "experimentation" / "chat-app" / "logs"
    assert log.log_dir() == expected
    assert (expected.parents[2] / "pyproject.toml").exists()


def test_setup_creates_the_folder_and_writes_a_timestamped_line(log_folder):
    logger = log.setup("probe")
    logger.info("hello %s", "world")
    text = read(log_folder, "probe")
    assert "INFO" in text and "tiny_agent.probe: hello world" in text
    assert text[:4].isdigit()  # starts with the year


def test_each_tool_gets_its_own_file(log_folder):
    log.setup("cli").info("from cli")
    log.setup("main").info("from main")
    assert "from cli" in read(log_folder, "cli") and "from main" not in read(log_folder, "cli")
    assert "from main" in read(log_folder, "main")


def test_calling_setup_twice_does_not_duplicate_lines(log_folder):
    log.setup("probe")
    logger = log.setup("probe")
    logger.info("once")
    assert read(log_folder, "probe").count("once") == 1
    assert len([h for h in logger.handlers if getattr(h, "_tiny_agent_file_handler", False)]) == 1


def test_setup_follows_a_changed_folder(log_folder, tmp_path, monkeypatch):
    log.setup("probe").info("first")
    other = tmp_path / "elsewhere"
    monkeypatch.setenv("TINYAGENT_LOG_DIR", str(other))
    log.setup("probe").info("second")
    assert "first" in read(log_folder, "probe") and "second" not in read(log_folder, "probe")
    assert "second" in read(other, "probe")


def test_nothing_reaches_the_terminal(log_folder, capsys):
    logger = log.setup("probe")
    logger.info("quiet")
    logger.error("also quiet")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_the_level_can_be_changed(log_folder, monkeypatch):
    monkeypatch.setenv("TINYAGENT_LOG_LEVEL", "WARNING")
    logger = log.setup("probe")
    logger.info("hidden")
    logger.warning("shown")
    text = read(log_folder, "probe")
    assert "hidden" not in text and "shown" in text


def test_an_unknown_level_falls_back_to_info(log_folder, monkeypatch):
    monkeypatch.setenv("TINYAGENT_LOG_LEVEL", "LOUD")
    logger = log.setup("probe")
    logger.info("kept")
    assert "kept" in read(log_folder, "probe")


def test_a_folder_that_cannot_be_created_never_stops_the_tool(tmp_path, monkeypatch):
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")
    monkeypatch.setenv("TINYAGENT_LOG_DIR", str(blocker / "logs"))  # cannot be a folder under a file
    logger = log.setup("probe")
    logger.info("dropped quietly")  # must not raise


def test_non_ascii_text_is_written_as_utf8(log_folder):
    log.setup("probe").info("héllo 🌍 日本語")
    assert "héllo 🌍 日本語" in read(log_folder, "probe")


def test_files_rotate_instead_of_growing_forever(log_folder):
    logger = log.setup("probe")
    handler = next(h for h in logger.handlers if getattr(h, "_tiny_agent_file_handler", False))
    assert (handler.maxBytes, handler.backupCount) == (1_000_000, 3)


# ---- what cli.py writes ---------------------------------------------------------------

def test_cli_logs_a_one_shot_question_and_answer(fake_server, log_folder):
    fake_server.router(a="loaded")
    fake_server.reply("42")
    assert cli.main(["--url", fake_server.url, "What is 6x7?"]) == 0
    text = read(log_folder, "cli")
    assert "tiny-agent started" in text
    assert "using model a" in text
    assert "one-shot question: What is 6x7?" in text
    assert "one-shot answer (m, 3 prompt + 4 completion tokens): 42" in text


def test_cli_logs_a_chat_session(fake_server, log_folder, monkeypatch):
    fake_server.router(a="loaded")
    fake_server.reply("hello back")
    lines = iter(["hi there", "/models", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["--url", fake_server.url]) == 0
    text = read(log_folder, "cli")
    assert "chat started: model=a" in text
    assert "user: hi there" in text
    assert "assistant (m, 3 prompt + 4 completion tokens): hello back" in text
    assert "command: /models" in text and "chat ended (/exit)" in text


def test_cli_logs_model_failures_as_errors(fake_server, log_folder):
    fake_server.router(a="loaded")
    fake_server.chat_status = 500
    fake_server.chat = {"error": {"message": "boom"}}
    assert cli.main(["--url", fake_server.url, "hi"]) == 1
    errors = [line for line in read(log_folder, "cli").splitlines() if " ERROR " in line]
    assert len(errors) == 1 and "model call failed" in errors[0] and "boom" in errors[0]


def test_cli_logs_when_no_model_can_be_used(dead_url, log_folder):
    assert cli.main(["--url", dead_url, "hi"]) == 2
    assert "ERROR   tiny_agent.cli: no usable model: Cannot reach the model server" in read(log_folder, "cli")


def test_cli_logs_a_model_substitution_as_a_warning(fake_server, log_folder):
    fake_server.router(a="loaded")
    fake_server.reply("ok")
    cli.main(["--url", fake_server.url, "-m", "ghost", "hi"])
    assert "WARNING tiny_agent.cli: 'ghost' is not offered here" in read(log_folder, "cli")


def test_cli_logs_a_cancelled_turn_and_an_empty_reply(fake_server, log_folder):
    fake_server.router(a="loaded")
    fake_server.reply(None)
    cli.chat(cli.LLM(model="a", base_url=fake_server.url), input_fn=feed(["hi", "/exit"]), output=lambda s: None)
    # chat() alone has no file handler (main() adds it), so nothing is written yet:
    assert not (log_folder / "cli.log").exists()
    log.setup("cli")
    cli.chat(cli.LLM(model="a", base_url=fake_server.url), input_fn=feed(["hi", "/exit"]), output=lambda s: None)
    assert "WARNING tiny_agent.cli: the model sent no text" in read(log_folder, "cli")


def test_chat_is_silent_on_the_terminal_without_main(fake_server, capsys):
    """Used on its own, chat() logs to nowhere: a warning must not leak to stderr."""
    fake_server.router(a="loaded")
    fake_server.chat_status = 500
    fake_server.chat = {"error": {"message": "boom"}}
    cli.chat(cli.LLM(model="a", base_url=fake_server.url), input_fn=feed(["hi", "/exit"]), output=lambda s: None)
    assert capsys.readouterr().err == ""


# ---- what main.py writes --------------------------------------------------------------

def test_main_logs_the_task_and_the_result(log_folder, demo_server, capsys):
    demo.main()
    text = read(log_folder, "main")
    assert "running task: Hello, agent!" in text
    assert "result: Received: Hello, agent!" in text
    assert capsys.readouterr().out.strip() == "Received: Hello, agent!"


def test_running_main_twice_appends(log_folder, demo_server):
    demo.main()
    demo.main()
    assert read(log_folder, "main").count("running task") == 2
