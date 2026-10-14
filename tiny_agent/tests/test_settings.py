"""Tests for ``Settings``: every TINYAGENT_* value, read by pydantic-settings from the environment and the .env file."""

import os

import pytest
from pydantic import ValidationError

from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.memory.trimming import TrimmingMemory
from tiny_agent.config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_URL,
    Settings,
    default_env_file,
)
from tiny_agent.main import memory_from_env


def write_env(tmp_path, text: str):
    """Write a settings file and point the code at it."""
    path = tmp_path / "test.env"
    path.write_text(text, encoding="utf-8")
    os.environ["TINYAGENT_ENV_FILE"] = str(path)
    return path


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
