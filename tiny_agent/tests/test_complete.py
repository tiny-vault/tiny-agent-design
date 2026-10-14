"""Integration scenarios: follow a real model request through the harness.

A successful arithmetic function alone does not prove that a model requested
it correctly. The happy-path stages trace definitions, calls, observations
and follow-up answers through the deployed server.

What this file demonstrates:
Tests for ``divide``, the complete happy path and the MMLU-Pro evaluation in ``tiny_agent/complete``.

The happy path runs against the real model in Docker (``node execute.mjs up llm-client``, then
``node execute.mjs llm-client load qwen3.5-4b``). There are no fakes: when no chat model is loaded, those tests are
skipped and say which command starts one. Chat replies vary, so they check for the tool called and the number.

Expected observations and limits:
Local division tests need no server. Happy-path and parameterized full-run
scenarios need their respective deployed chat or embedding endpoints and skip
on ``StackNotRunning``. The full-run approval scenario scripts input with
``patch``; model calls remain real. Keyword checks illustrate expected facts,
not robust grading of arbitrary natural-language answers.
"""

import pytest

from tiny_agent.complete.evaluation import run_evaluation
from tiny_agent.complete.full_run import SCENARIOS
from tiny_agent.complete.happy_path import (
    StackNotRunning,
    learned_stage,
    prompting_stage,
    real_llm,
    run_happy_path,
)
from tiny_agent.components.benchmarks import ifeval
from tiny_agent.components.tools.functions.divide import divide


def test_divide_takes_text_and_returns_a_float():
    assert divide("42", "6") == 7.0


def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        divide("1", "0")


@pytest.fixture(scope="module")
def llm():
    """The real chat model, or a skip when none is loaded."""
    try:
        return real_llm()
    except StackNotRunning as error:
        pytest.skip(str(error))


def test_prompting_stage_multiplies(llm):
    stage = prompting_stage(llm)
    assert stage["action"]["tool"] == "multiply"
    assert stage["observation"] == "OBSERVATION: 42.0"


def test_learned_stage_divides(llm):
    stage = learned_stage(llm, "OBSERVATION: 42.0")
    assert stage["action"]["tool"] == "divide"
    assert stage["observation"] == "OBSERVATION: 7.0"


def test_happy_path_runs_prompting_then_learned(llm):
    stages = run_happy_path(llm)
    assert list(stages) == ["prompting", "learned"]
    assert "7" in stages["learned"]["answer"]


def test_evaluation_scores_every_mmlu_pro_question(llm):
    report = run_evaluation(llm)
    # The model's letters vary, so check the shape and that each reply was scored, not the pass rate itself
    assert report["name"] == "MMLU Pro" and len(report["results"]) == 3
    assert all(isinstance(result["passed"], bool) for result in report["results"])
    assert report["pass_rate"] == sum(result["passed"] for result in report["results"]) / 3


def test_evaluation_checks_every_ifeval_instruction(llm):
    report = run_evaluation(llm, ifeval)
    # Whether the model obeys varies, so check that each reply was written and checked
    assert report["name"] == "IFeval" and len(report["results"]) == 3
    assert all(result["prediction"] and isinstance(result["passed"], bool) for result in report["results"])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_full_run_scenario(name):
    """Each part of the harness, run for real (see ``complete/full_run.py``); skipped when a server is down."""
    try:
        SCENARIOS[name]()
    except StackNotRunning as error:
        pytest.skip(str(error))
