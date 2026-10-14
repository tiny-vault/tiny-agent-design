"""Evaluate TinyAgent on the real Docker model: three MMLU-Pro questions, then three IFEval prompts, a fresh agent for each.

What is it? ``Evaluator`` runs an agent over a ``Benchmark`` and scores every reply; this is that run against the
model server of ``node execute.mjs up llm-client`` (``TINYAGENT_LLM_URL``, default ``http://127.0.0.1:3200/v1``;
load a chat model with ``node execute.mjs llm-client load qwen3.5-4b``). Nothing is faked.

``create_agent`` builds the agent the book evaluates: an empty ``Memory``, an empty ``LearnedTools`` registry (the
book's ``NativeTools``) and ``NativeReAct``, so the model answers each multiple-choice question from what it knows,
with no tools. The evaluator calls it once per question, so no question sees another's conversation.
``exact_match_scorer`` reads the first standalone letter of each reply and compares it with the correct one.
IFEval is the same agent on a different kind of test: no right answer, only an instruction to follow (fewer than 10
sentences, at most 150 words, no commas), which ``programmatic_scorer`` checks with each example's own ``check``.

Each result is the evaluator's report: ``{"name": "MMLU Pro", "pass_rate": ..., "results": [...]}`` with each
example's ``prediction`` and whether it ``passed``. Replies vary between runs, so the pass rate can too; three
questions are a demonstration, not a measurement.

Run it with ``python -m tiny_agent.complete.evaluation``.
"""

import sys
from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.complete.happy_path import StackNotRunning, real_llm
from tiny_agent.components.benchmarks import ifeval, mmlu_pro
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.evaluator import Benchmark, Evaluator
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools


def run_evaluation(llm: LLM | None = None, benchmark: Benchmark = mmlu_pro) -> dict[str, Any]:
    """Run ``benchmark`` with a fresh agent per example and return the evaluator's report.

    Args:
        llm: The model. Defaults to :func:`~tiny_agent.complete.happy_path.real_llm`.
        benchmark: What to run: ``mmlu_pro`` (the default) or ``ifeval``.

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
    return Evaluator(create_agent).run(benchmark)


def main() -> int:
    """Run both evaluations, print each report, and return the exit code."""
    try:
        llm = real_llm()
        for benchmark in (mmlu_pro, ifeval):
            result = run_evaluation(llm, benchmark)
            print(result)
    except StackNotRunning as error:
        print(f"cannot run: {error}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
