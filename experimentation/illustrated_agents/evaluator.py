"""Evaluating ``tiny_agent``: run a fresh agent over a small benchmark and get a pass rate.

**What it shows.** One good answer says little about an agent; a benchmark makes it measurable. A ``Benchmark`` is a name, a list of examples (each a ``task`` and the expected ``answer``) and a scorer that grades a prediction against its example. ``Evaluator`` runs every task and averages the scores into a ``pass_rate``.

**How it works here.** Everything comes from the repo's ``tiny_agent`` package. ``create_agent`` is the book's factory: a new ``TinyAgent`` with an empty ``Memory``, an empty ``LearnedTools`` registry (the book's ``NativeTools``) and ``NativeReAct``, so the model answers from its own knowledge with no tools. The evaluator calls it once per example, so no task sees another's conversation. ``benchmark`` holds four arithmetic ``examples``; ``contains_answer`` passes a prediction when the expected number appears in it, so "The answer is 42." counts. ``show_report`` prints each prediction with its result, then the pass rate.

**What to expect.** Each example is a full agent run, so this takes a few seconds per task. A 4B model usually gets most of these right, but the pass rate can change between runs, which is exactly why a benchmark runs many examples. ``contains_answer`` is lenient: a reply that mentions the number in passing also passes. ``llm`` is ``qwen3.5-4b`` via the gateway, without thinking. Compare ``native_react``, the agent on one task, and ``tiny_agent/components/evaluator.py``.
"""

from pathlib import Path

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.evaluator import Benchmark, Evaluator
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

# The examples: a task for the agent and the answer the scorer looks for
examples = [
    {"task": "What is 6 times 7? Answer with the number.", "answer": "42"},
    {"task": "What is 15 plus 27? Answer with the number.", "answer": "42"},
    {"task": "What is 100 minus 58? Answer with the number.", "answer": "42"},
    {"task": "What is 144 divided by 12? Answer with the number.", "answer": "12"},
]


def contains_answer(prediction, example):
    """Pass when the expected answer appears anywhere in the prediction (lenient: free wording is fine)."""
    return example["answer"] in prediction


benchmark = Benchmark(name="arithmetic", examples=examples, scorer=contains_answer)


def create_agent():
    """Create a new instance of TinyAgent"""
    return TinyAgent(
        llm=llm,
        memory=Memory(),
        tools=LearnedTools(),
        planner=NativeReAct(),
    )


def show_report(report):
    """Print each example's prediction and whether it passed, then the pass rate."""
    for example, result in zip(benchmark.examples, report["results"]):
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{mark}] {example['task']}\n       -> {result['prediction'].strip()!r} (expected {example['answer']})")
    print(f"\n{report['name']}: pass rate {report['pass_rate']:.0%}")


if __name__ == "__main__":
    # Needs the Docker stack with qwen3.5-4b loaded
    evaluator = Evaluator(create_agent)
    show_report(evaluator.run(benchmark))
