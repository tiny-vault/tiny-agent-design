"""Tests for evaluation: a benchmark holds its examples and scorer, the evaluator runs a fresh agent per example."""

from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.response import Response
from tiny_agent.components.evaluator import Benchmark, Evaluator, Scorer
from tiny_agent.components.memory.memory import Memory
from tiny_agent.tests.stubs import ScriptedLLM


def exact(prediction: str, example: dict[str, Any]) -> bool:
    return prediction.strip() == example["answer"]


def keywords(prediction: str, example: dict[str, Any]) -> float:
    found = [word for word in example["keywords"] if word in prediction]
    return len(found) / len(example["keywords"])


def test_benchmark_keeps_its_name_examples_and_scorer():
    examples = [{"task": "What is 6 times 7?", "answer": "42"}]
    benchmark = Benchmark(name="arithmetic", examples=examples, scorer=exact)
    assert (benchmark.name, benchmark.examples, benchmark.scorer) == ("arithmetic", examples, exact)


def test_a_scorer_returns_a_pass_or_a_partial_score():
    scorers: list[Scorer] = [exact, keywords]
    example = {"task": "t", "answer": "42", "keywords": ["42", "six"]}
    assert [scorer("42", example) for scorer in scorers] == [True, 0.5]


def test_evaluator_scores_each_example_and_averages_a_pass_rate():
    llm = ScriptedLLM("42", "5", Response(content=None))
    benchmark = Benchmark(
        "arithmetic",
        [{"task": "6 x 7?", "answer": "42"}, {"task": "2 + 2?", "answer": "4"}, {"task": "1 + 0?", "answer": "1"}],
        exact,
    )
    report = Evaluator(lambda: TinyAgent(llm, Memory())).run(benchmark)
    assert (report["name"], report["pass_rate"]) == ("arithmetic", 1 / 3)
    # A reply with no text is scored as ""
    assert report["results"] == [
        {"prediction": "42", "passed": True},
        {"prediction": "5", "passed": False},
        {"prediction": "", "passed": False},
    ]


def test_evaluator_builds_a_fresh_agent_per_example():
    llm = ScriptedLLM("a", "b")
    agents: list[TinyAgent] = []

    def create_agent() -> TinyAgent:
        agents.append(TinyAgent(llm, Memory()))
        return agents[-1]

    Evaluator(create_agent).run(Benchmark("two", [{"task": "first"}, {"task": "second"}], lambda p, e: True))
    # The second agent never saw the first task: no memory leaks between examples
    assert len(agents) == 2
    assert [m["content"] for m in llm.calls[1]] == ["second"]


def test_evaluator_with_partial_scores_and_no_examples():
    llm = ScriptedLLM("42 six", "42")
    examples = [{"task": "t", "keywords": ["42", "six"]}] * 2
    assert Evaluator(lambda: TinyAgent(llm, Memory())).run(Benchmark("kw", examples, keywords))["pass_rate"] == 0.75
    assert Evaluator(lambda: TinyAgent(llm, Memory())).run(Benchmark("empty", [], exact))["pass_rate"] == 0.0
