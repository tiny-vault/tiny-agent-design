"""Tests for ``experimentation/trigger_ias.py`` and the scripts it runs.

The first group is offline. The second runs every script for real as a subprocess against the Docker stack and is skipped unless the stack is up with ``qwen3.5-4b`` loaded. Run them with ``node execute.mjs test ias``.
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRIGGER = ROOT / "experimentation" / "trigger_ias.py"
KEY_FILE = ROOT / "experimentation" / "chat-app" / "secrets" / "llama_api_key"
MODEL = "qwen3.5-4b"

sys.path.insert(0, str(TRIGGER.parent))
import trigger_ias  # noqa: E402

# Scripts that sample many answers (10 candidates plus judge calls) or run six prompts
SLOW = {name for name in trigger_ias.scripts() if name.startswith("best_of_n/") or name.endswith("/compare")}


def test_scripts_found_in_folders():
    names = trigger_ias.scripts()
    assert "agent_without_memory" in names
    assert "best_of_n/orm" in names and "shot_prompting/zero_shot" in names
    # No helper or test module is offered as a runnable script
    assert not any("tests" in name or name == "__init__" for name in names)


def test_list_and_unknown_name_exit_codes():
    listed = subprocess.run([sys.executable, str(TRIGGER), "--list"], capture_output=True, text=True, cwd=ROOT)
    assert listed.returncode == 0 and "best_of_n/orm" in listed.stdout
    unknown = subprocess.run([sys.executable, str(TRIGGER), "nope"], capture_output=True, text=True, cwd=ROOT)
    assert unknown.returncode == 2 and "Unknown script" in unknown.stdout


def test_terminal_choice_by_number_and_name(monkeypatch):
    names = ["a", "b/c"]
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert trigger_ias.terminal_choice(names) == "b/c"
    monkeypatch.setattr("builtins.input", lambda _: "a.py")
    assert trigger_ias.terminal_choice(names) == "a"
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert trigger_ias.terminal_choice(names) is None


def test_repl_runs_choice_then_survives_a_failing_script(monkeypatch, capsys):
    ran = []
    answers = iter(["boom", "ok", None])

    def fake_run(name):
        ran.append(name)
        if name == "boom":
            raise RuntimeError("script failed")

    monkeypatch.setattr(trigger_ias, "run", fake_run)
    monkeypatch.setattr(trigger_ias, "popup_choice", lambda names: next(answers))
    assert trigger_ias.repl(["boom", "ok"]) == 0
    # The failing script did not end the loop
    assert ran == ["boom", "ok"]
    assert "script failed" in capsys.readouterr().err


def _model_loaded():
    """True if the stack answers and ``MODEL`` is loaded; False if it is down, locked or not loaded."""
    try:
        request = urllib.request.Request(
            "http://localhost:3200/v1/models", headers={"Authorization": f"Bearer {KEY_FILE.read_text().strip()}"}
        )
        with urllib.request.urlopen(request, timeout=5) as reply:
            models = json.load(reply)["data"]
    except Exception:
        return False
    return any(m.get("id") == MODEL and m.get("status", {}).get("value") == "loaded" for m in models)


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in trigger_ias.scripts()],
)
def test_script_runs_against_the_stack(name):
    if not _model_loaded():
        pytest.skip(f"stack down or {MODEL} not loaded (node execute.mjs up; node execute.mjs load {MODEL})")
    # Through trigger_ias.py, so the folder/script naming and the import path are tested too
    result = subprocess.run(
        [sys.executable, str(TRIGGER), name], capture_output=True, text=True, cwd=ROOT, timeout=1800
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip(), "the script printed nothing"
