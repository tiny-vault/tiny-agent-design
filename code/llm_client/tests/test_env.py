"""Tests for llm_client.env: loading settings from a .env file."""

import os

import pytest

from llm_client import env


def write(path, text, encoding="utf-8"):
    """Write a settings file and return its path."""
    path.write_text(text, encoding=encoding)
    return path


# ---- parse ----------------------------------------------------------------------------

def test_plain_name_value_lines():
    assert env.parse("A=1\nB=two words\n") == {"A": "1", "B": "two words"}


def test_blank_lines_and_comments_are_skipped():
    assert env.parse("\n# a comment\n   \nA=1\n  # indented comment\n") == {"A": "1"}


def test_an_empty_value_is_an_empty_string():
    assert env.parse("A=\nB=   \n") == {"A": "", "B": ""}


def test_spaces_around_the_equals_sign_and_the_line_are_trimmed():
    assert env.parse("  A  =  value  \n") == {"A": "value"}


def test_export_prefix_is_accepted():
    assert env.parse("export A=1\nexport   B=2\n") == {"A": "1", "B": "2"}


def test_double_quotes_keep_spaces_and_hashes():
    assert env.parse('A="  spaced # not a comment  "\n') == {"A": "  spaced # not a comment  "}


def test_double_quotes_understand_escapes():
    assert env.parse(r'A="line one\nline two\t\"quoted\" back\\slash"') == {"A": 'line one\nline two\t"quoted" back\\slash'}


def test_an_unknown_escape_is_kept_as_written():
    assert env.parse(r'A="a\qb"') == {"A": "a\\qb"}


def test_single_quotes_are_literal():
    assert env.parse(r"A='no \n escapes # here'") == {"A": r"no \n escapes # here"}


def test_text_after_a_closing_quote_is_ignored():
    assert env.parse('A="value" # comment\nB=\'x\' trailing\n') == {"A": "value", "B": "x"}


def test_an_unclosed_quote_takes_the_rest_of_the_line():
    assert env.parse('A="never closed\nB=\'also open\n') == {"A": "never closed", "B": "also open"}


def test_a_trailing_comment_needs_a_space_before_the_hash():
    assert env.parse("A=value # comment\nB=a#b\nC=value\t# tab comment\n") == {"A": "value", "B": "a#b", "C": "value"}


def test_the_value_may_contain_equals_signs():
    assert env.parse("URL=http://x/?a=1&b=2\n") == {"URL": "http://x/?a=1&b=2"}


def test_the_last_duplicate_wins():
    assert env.parse("A=1\nA=2\n") == {"A": "2"}


@pytest.mark.parametrize("line", ["no equals sign", "=novalue", "1BAD=x", "BAD-NAME=x", "BAD NAME=x", "export", "export =x"])
def test_lines_that_cannot_be_understood_are_skipped(line):
    assert env.parse(line + "\nGOOD=1\n") == {"GOOD": "1"}


def test_windows_line_endings():
    assert env.parse("A=1\r\nB=2\r\n") == {"A": "1", "B": "2"}


def test_unicode_values():
    assert env.parse("A=héllo 🌍 日本語\n") == {"A": "héllo 🌍 日本語"}


def test_empty_text():
    assert env.parse("") == {}


# ---- load_env -------------------------------------------------------------------------

def test_load_sets_missing_names_and_reports_them(tmp_path):
    path = write(tmp_path / ".env", "TINYAGENT_MODEL=m\nTINYAGENT_THINK=1\n")
    assert env.load_env(path) == {"TINYAGENT_MODEL": "m", "TINYAGENT_THINK": "1"}
    assert os.environ["TINYAGENT_MODEL"] == "m" and os.environ["TINYAGENT_THINK"] == "1"


def test_the_real_environment_wins_by_default(tmp_path):
    os.environ["TINYAGENT_MODEL"] = "from-the-shell"
    path = write(tmp_path / ".env", "TINYAGENT_MODEL=from-the-file\nTINYAGENT_THINK=1\n")
    assert env.load_env(path) == {"TINYAGENT_THINK": "1"}  # only what was missing
    assert os.environ["TINYAGENT_MODEL"] == "from-the-shell"


def test_override_lets_the_file_win(tmp_path):
    os.environ["TINYAGENT_MODEL"] = "from-the-shell"
    path = write(tmp_path / ".env", "TINYAGENT_MODEL=from-the-file\n")
    assert env.load_env(path, override=True) == {"TINYAGENT_MODEL": "from-the-file"}
    assert os.environ["TINYAGENT_MODEL"] == "from-the-file"


def test_a_name_set_to_an_empty_string_in_the_environment_still_wins(tmp_path):
    os.environ["TINYAGENT_MODEL"] = ""
    write(tmp_path / ".env", "TINYAGENT_MODEL=from-the-file\n")
    assert env.load_env(tmp_path / ".env") == {}
    assert os.environ["TINYAGENT_MODEL"] == ""


def test_a_missing_file_is_not_an_error(tmp_path):
    assert env.load_env(tmp_path / "nope.env") == {}


def test_an_empty_file_loads_nothing(tmp_path):
    assert env.load_env(write(tmp_path / ".env", "")) == {}


def test_a_folder_instead_of_a_file_is_not_an_error(tmp_path):
    assert env.load_env(tmp_path) == {}


def test_a_file_that_is_not_text_is_not_an_error(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"\xff\xfe\x00A=1\x80\x81")
    assert env.load_env(path) == {}


def test_a_byte_order_mark_is_ignored(tmp_path):
    path = write(tmp_path / ".env", "TINYAGENT_MODEL=m\n", encoding="utf-8-sig")
    assert env.load_env(path) == {"TINYAGENT_MODEL": "m"}


def test_garbage_lines_do_not_stop_the_good_ones(tmp_path):
    path = write(tmp_path / ".env", "???\nTINYAGENT_MODEL=m\n=\n")
    assert env.load_env(path) == {"TINYAGENT_MODEL": "m"}


def test_loading_twice_is_harmless(tmp_path):
    path = write(tmp_path / ".env", "TINYAGENT_MODEL=m\n")
    env.load_env(path)
    assert env.load_env(path) == {}  # already set the first time
    assert os.environ["TINYAGENT_MODEL"] == "m"


# ---- which file is read ------------------------------------------------------------------

def test_the_default_file_is_code_dot_env():
    os.environ.pop("TINYAGENT_ENV_FILE")
    assert env.default_path().name == ".env"
    assert env.default_path().parent.name == "code"


def test_tinyagent_env_file_chooses_the_file(tmp_path):
    path = write(tmp_path / "custom.env", "TINYAGENT_MODEL=custom\n")
    os.environ["TINYAGENT_ENV_FILE"] = str(path)
    assert env.default_path() == path
    assert env.load_env() == {"TINYAGENT_MODEL": "custom"}


def test_the_real_dot_env_in_the_repo_loads_without_error():
    """Whatever is in code/.env (even nothing), loading it must work."""
    os.environ.pop("TINYAGENT_ENV_FILE")
    assert isinstance(env.load_env(), dict)
