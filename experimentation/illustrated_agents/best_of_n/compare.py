"""Compare the five Best-of-N pickers (unit tests, ORM, PRM, majority vote, weighted PRM) on the same 10 ``roman_to_int`` candidates.

**What it is.** Best-of-N draws N answers from a model and keeps the best one according to a verifier. The model never improves; the selection does, so the verifier decides everything. Each picker has its own script (``unit_tests``, ``orm``, ``prm``, ``majority_vote``, ``weighted_prm``) that samples its own candidates; this script instead samples **once** and lets every picker choose from the same pool, so the picks are directly comparable.

**How it works here.** ``compare(n)`` samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1``), then scores each with ``verify`` (unit tests, the ground truth), ``orm_score`` (0 to 1, whole answer), ``prm_scores`` (the weakest step, ``min``) and ``behaviour`` (the answer it votes for). Four pickers choose from those columns: the top ORM score, the top PRM score, the largest ``behaviour`` group, and the group with the largest summed PRM reward. ``judge`` (the same model at ``temperature=0``) stands in for both reward models, so their scores are the model's opinion, not a measured quality. The final table shows each pick's unit-test score next to the best possible one.

**Running it.** ``python compare.py``. One sample of 10 per run, so one run does not rank the verifiers; it only shows how they can disagree.

Warning:
    ``verify`` and ``behaviour`` run model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

import re
from collections import defaultdict
from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[2] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)
# The reward model: same model, greedy, so one candidate always gets the same score.
judge = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=0)

query = """Write a Python function `roman_to_int(s)` that converts a Roman
    numeral
string to an integer. The function should:
- Handle standard Roman numerals from 1 to 3999.
- Correctly handle subtractive notation (e.g., IV=4, IX=9, XL=40, XC=90, CD=400,
CM=900).
- Return 0 for invalid inputs (empty string, non-Roman characters, or invalid
patterns
  like "IIII" or "VV").

Give only the function definition, no explanation or markdown formatting."""

ORM_PROMPT = """You are an outcome reward model. Judge only whether the final solution fully solves the task.

Task:
{task}

Solution:
{solution}

Rate how likely the solution is completely correct on a scale from 0 to 10. Reply with only the number."""

PRM_PROMPT = """You are a process reward model. Judge each step of a partial solution, in order.

Task:
{task}

Solution, one numbered step per line:
{steps}

For every step, rate from 0 to 1 how likely it is correct and moves toward a correct solution, given the steps before it. Reply with exactly one line per step in the form `<step number>: <score>` and nothing else."""

# Ground truth for verify; the reward models never see it, and behaviour uses only the inputs
test_cases = [
    ("III", 3),
    ("IV", 4),
    ("IX", 9),
    ("LVIII", 58),
    ("MCMXCIV", 1994),
    ("MMMCMXCIX", 3999),
    ("", 0),
    ("IIII", 0),       # invalid: four in a row
    ("VV", 0),         # invalid: V can't repeat
    ("IC", 0),         # invalid: I can only subtract from V or X
    ("ABC", 0),        # invalid: non-Roman characters
    ("MMMM", 0),       # invalid: exceeds 3999
]


def verify(answer):
    """Score a candidate by the fraction of ``test_cases`` it passes.

    Args:
        answer: The model's reply, expected to be plain Python defining ``roman_to_int``.

    Returns:
        A score from 0.0 to 1.0; 0.0 if the code does not run, lacks ``roman_to_int`` or crashes.

    Warning:
        Runs model-written code with ``exec`` and no sandbox.
    """
    try:
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]
        # (the book's listing wraps `len(test_cases)` across two lines; as printed, `/ len` divides by the
        # builtin, raises TypeError and every candidate scores 0.0, so it is joined here)
        return sum(fn(inp) == expected for inp, expected in test_cases) / len(test_cases)
    except Exception:
        # Invalid Python, a missing function or a crash on any input scores 0
        return 0.0


def orm_score(answer):
    """Ask the judge for an outcome reward: one 0 to 10 rating of the whole answer, scaled to 0.0 to 1.0.

    Args:
        answer: A finished candidate solution.

    Returns:
        The first number in the judge's reply, capped at 10 and divided by 10; 0.0 if there is none.
    """
    # One judge call per candidate: the whole answer in, a single score out (unparseable scores 0)
    reply = judge.generate([{"role": "user", "content": ORM_PROMPT.format(task=query, solution=answer)}]).content or ""
    match = re.search(r"\d+(?:\.\d+)?", reply)
    return min(float(match.group()), 10.0) / 10 if match else 0.0


def split_steps(answer):
    """Split a candidate into the "steps" the process reward model scores: its non-empty code lines."""
    # A step is one non-empty line of the candidate's code
    return [line for line in answer.splitlines() if line.strip()]


def prm_scores(steps):
    """Ask the judge for a process reward: one 0 to 1 score per step, from a single call.

    Args:
        steps: Lines of code from ``split_steps``.

    Returns:
        One score per step, in order; a step the judge did not score gets 0.0.
    """
    # One judge call per candidate: all numbered steps in, one score per step out (a missing step scores 0)
    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    reply = judge.generate([{"role": "user", "content": PRM_PROMPT.format(task=query, steps=numbered)}]).content or ""
    found = {int(n): min(float(s), 1.0) for n, s in re.findall(r"^\s*(\d+)\s*[:.)-]\s*(\d+(?:\.\d+)?)", reply, re.M)}
    return [found.get(i, 0.0) for i in range(1, len(steps) + 1)]


def behaviour(answer):
    """Fingerprint a candidate by its outputs on the test inputs, so equivalent answers can be grouped.

    Args:
        answer: The model's reply, expected to define ``roman_to_int``.

    Returns:
        A string of the outputs (``error`` for an input that raises), or ``does not run`` if the code fails to load.

    Warning:
        Runs model-written code with ``exec`` and no sandbox.
    """
    # The candidate's outputs on the inputs; two candidates with the same outputs count as the same answer
    try:
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]
    except Exception:
        return "does not run"
    outputs = []
    for inp, _ in test_cases:
        try:
            outputs.append(repr(fn(inp)))
        except Exception:
            outputs.append("error")
    return ", ".join(outputs)


def sample():
    """Draw one candidate: a fresh reply to ``query`` from the temperature-1 ``llm``."""
    return llm.generate([{"role": "user", "content": query}]).content


def compare(n=10):
    """Sample ``n`` candidates once, let all five pickers choose from them and print a table of the picks.

    Args:
        n: Number of candidates to sample.

    Returns:
        A dict from picker name to the unit-test score (``verify``) of the candidate it picked.
    """
    # Score every candidate on every axis once; each picker then reads only the columns it is allowed to
    candidates = []
    for i in range(n):
        answer = sample()
        orm = orm_score(answer)
        prm = min(prm_scores(split_steps(answer)) or [0.0])
        candidates.append({"answer": answer, "tests": verify(answer), "orm": orm, "prm": prm, "behaviour": behaviour(answer)})

    print(f"{'cand':<6}{'unit tests':>11}{'ORM':>7}{'PRM':>7}")
    for i, c in enumerate(candidates, 1):
        print(f"{i:<6}{c['tests']:>11.2f}{c['orm']:>7.2f}{c['prm']:>7.2f}")

    # Votes and PRM weight per answer (behaviour group)
    votes, weight = defaultdict(list), defaultdict(float)
    for c in candidates:
        votes[c["behaviour"]].append(c)
        weight[c["behaviour"]] += c["prm"]

    picks = {
        "unit_tests": max(candidates, key=lambda c: c["tests"]),
        "orm": max(candidates, key=lambda c: c["orm"]),
        "prm": max(candidates, key=lambda c: c["prm"]),
        # A tie in votes goes to the group seen first
        "majority_vote": votes[max(votes, key=lambda k: len(votes[k]))][0],
        "weighted_prm": max(votes[max(weight, key=weight.get)], key=lambda c: c["prm"]),
    }

    best = max(c["tests"] for c in candidates)
    print(f"\n{'picker':<16}{'unit tests of pick':>20}{'best possible':>16}")
    for name, pick in picks.items():
        print(f"{name:<16}{pick['tests']:>20.2f}{best:>16.2f}")
    return {name: pick["tests"] for name, pick in picks.items()}


if __name__ == "__main__":
    # Needs the Docker stack with qwen3.5-4b loaded
    compare()
