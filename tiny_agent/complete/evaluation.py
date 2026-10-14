"""Evaluate TinyAgent on the real Docker model: the three MMLU-Pro questions, a fresh agent for each.

What is it? ``Evaluator`` runs an agent over a ``Benchmark`` and scores every reply; this is that run against the
model server of ``node execute.mjs up llm-client`` (``TINYAGENT_LLM_URL``, default ``http://127.0.0.1:3200/v1``;
load a chat model with ``node execute.mjs llm-client load qwen3.5-4b``). Nothing is faked.

``create_agent`` builds the agent the book evaluates: an empty ``Memory``, an empty ``LearnedTools`` registry (the
book's ``NativeTools``) and ``NativeReAct``, so the model answers each multiple-choice question from what it knows,
with no tools. The evaluator calls it once per question, so no question sees another's conversation.
``exact_match_scorer`` reads the first standalone letter of each reply and compares it with the correct one.

The result is the evaluator's report: ``{"name": "MMLU Pro", "pass_rate": ..., "results": [...]}`` with each
question's ``prediction`` and whether it ``passed``. Replies vary between runs, so the pass rate can too; three
questions are a demonstration, not a measurement.

Run it with ``python -m tiny_agent.complete.evaluation``.
"""

import sys
from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.complete.happy_path import StackNotRunning, real_llm
from tiny_agent.components.benchmarks import mmlu_pro
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.evaluator import Evaluator
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools


def run_evaluation(llm: LLM | None = None) -> dict[str, Any]:
    """Run ``mmlu_pro`` with a fresh agent per question and return the evaluator's report.

    Args:
        llm: The model. Defaults to :func:`~tiny_agent.complete.happy_path.real_llm`.

    Returns:
        ``{"name", "pass_rate", "results"}`` from :meth:`~tiny_agent.components.evaluator.Evaluator.run`.

    Raises:
        StackNotRunning: There is no model to talk to.
    """
    model = llm or real_llm()

    def create_agent() -> TinyAgent:
        """Create a new instance of TinyAgent"""
        return TinyAgent(
            llm=model,
            memory=Memory(),
            tools=LearnedTools(),
            planner=NativeReAct(),
        )

    # Run evaluation
    return Evaluator(create_agent).run(mmlu_pro)


def main() -> int:
    """Run the evaluation, print the report, and return the exit code."""
    try:
        result = run_evaluation()
    except StackNotRunning as error:
        print(f"cannot run: {error}")
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
