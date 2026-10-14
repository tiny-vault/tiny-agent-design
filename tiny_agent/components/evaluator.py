"""Evaluation: a fixed set of tasks with a rule for scoring the agent's answers.

One good run says little about an agent: the model is not deterministic and a
change to a prompt, a tool or a memory can fix one task and break another. A
*benchmark* makes that measurable. It is a named list of examples (each a dict
holding at least the ``task`` to give the agent and whatever the scorer needs,
such as the expected ``answer``) and a *scorer*: a function that takes the
agent's prediction and its example and says how good the prediction is.

A ``Scorer`` returns ``bool`` (pass or fail, such as an exact match) or
``float`` (a partial score, such as 0.5 for half the keywords found); ``True``
counts as 1 and ``False`` as 0, so either kind averages into a pass rate.
``exact_match_scorer`` is one for multiple-choice benchmarks (options A to J,
as in MMLU-Pro): it reads the first standalone letter of the prediction.

``Evaluator`` runs the agent over a benchmark. It is given a function that
builds a *fresh* agent, and calls it once per example, so no example sees the
memory of another (one shared agent would let early answers leak into later
ones). Each prediction is scored, and the scores are averaged into
``pass_rate``. A ``None`` reply counts as the empty string.

The evaluator does not check that the examples have the keys the scorer reads,
and it does not catch errors: a failing model call or scorer stops the whole
run. Every example is a full agent run, so a benchmark against a real model is
slow. Compare ``trajectory.py`` for what one run records.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tiny_agent.agent import TinyAgent

# Type hint for the scorers
##  (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict[str, Any]], bool | float]


def exact_match_scorer(prediction: str, example: dict[str, Any]) -> bool:
    """Return True if the answer matches the prediction, False otherwise.

    The chosen option is the first letter A to J that stands alone in the prediction (case is ignored),
    so ``"b"``, ``"B)"`` and ``"The answer is (B)"`` all read as ``B``. A heuristic: a standalone ``A``
    or ``I`` that is a word ("I think it is B") is read first, and the letter is not checked against the
    options the question offered.

    Args:
        prediction: The agent's reply.
        example: The benchmark example; ``example["expected"]`` is the correct letter, in upper case.

    Returns:
        Whether the first standalone letter equals ``expected``; ``False`` when there is no such letter.
    """
    match = re.search(r"\b([A-J])\b", prediction.upper())
    # No letter at all (an empty or free-text reply) is a wrong answer, not a crash that stops the evaluator
    return match is not None and match.group(1) == example["expected"]


@dataclass
class Benchmark:
    """A named set of examples and the scorer that grades a prediction for each.

    Attributes:
        name: What the benchmark measures, used to label its results.
        examples: The test cases. Each is a dict with the ``task`` for the agent and
            the fields the scorer compares against, for example ``answer``.
        scorer: A ``Scorer``: ``scorer(prediction, example)`` returns ``True`` or
            ``False``, or a float score.
    """

    name: str
    examples: list[dict[str, Any]]
    scorer: Scorer


class Evaluator:
    """Run a TinyAgent over a Benchmark and aggregate the results."""

    def __init__(self, create_agent: Callable[[], TinyAgent]):
        """Initialize with a function that creates a new agent instance.

        Args:
            create_agent: Called with no arguments once per example; it must return a new
                agent each time, so examples do not share memory.
        """
        self.create_agent = create_agent

    def run(self, benchmark: Benchmark) -> dict[str, Any]:
        """Run the agent on examples in the benchmark and score the results.

        Args:
            benchmark: The examples to run (each needs a ``task``) and the scorer to grade them.

        Returns:
            ``{"name": ..., "pass_rate": ..., "results": [...]}``: the benchmark's name, the mean
            score (``0.0`` for no examples) and, per example in order, its ``prediction`` and
            ``passed`` (the scorer's result).
        """

        # Run each example and collect results
        results: list[dict[str, Any]] = []
        for example in benchmark.examples:
            # A new agent per example: a shared memory would leak earlier answers into later tasks
            agent = self.create_agent()
            prediction = agent.run(example["task"]) or ""
            passed = benchmark.scorer(prediction, example)
            results.append(
                {
                    "prediction": prediction,
                    "passed": passed,
                }
            )

        # Aggregate pass rate (True counts as 1, False as 0, a float score as itself)
        if results:
            pass_rate = sum(
              result["passed"] for result in results
            ) / len(results)
        else:
            pass_rate = 0.0

        # Return detailed results and overall pass rate
        return {
            "name": benchmark.name,
            "pass_rate": pass_rate,
            "results": results,
        }
