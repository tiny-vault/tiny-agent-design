"""Weighted Best-of-N: sample 10 ``roman_to_int`` candidates, pool the PRM rewards of those that agree and keep the heaviest answer.

**What it is.** Weighted Best-of-N is a majority vote in which each vote is weighted by a reward. Candidates that return the same outputs on the test inputs (``behaviour``) form one answer; the answer's weight is the sum of its candidates' process-reward (PRM) scores. An answer many good samples agree on beats one lucky outlier.

**How it works here.** ``best_of_n_weighted_prm(n)`` samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1`` so candidates differ). ``split_steps`` and ``prm_scores`` ask ``judge`` (the same model at ``temperature=0``, standing in for a trained reward model) for a 0 to 1 score per code line with ``PRM_PROMPT``; a candidate's reward is its weakest step (``min``). The heaviest answer wins and its best-scored candidate is returned. The scores are the model's opinion, so ``verify`` also runs the unit tests afterwards, only to show whether the pick really is the best.

**Running it.** ``python weighted_prm.py``. Scores vary between runs. Compare ``majority_vote`` (every vote counts one) and ``prm`` (no voting, best single candidate), and ``compare`` for all five verifiers on the same candidates.

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

PRM_PROMPT = """You are a process reward model. Judge each step of a partial solution, in order.

Task:
{task}

Solution, one numbered step per line:
{steps}

For every step, rate from 0 to 1 how likely it is correct and moves toward a correct solution, given the steps before it. Reply with exactly one line per step in the form `<step number>: <score>` and nothing else."""

# Ground truth for verify; the reward model never sees it, and behaviour uses only the inputs
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

    Here it only reports how good the weighted pick was; it does not choose.

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


def best_of_n_weighted_prm(n=10):
    """Weighted Best-of-N: candidates with identical behaviour pool their PRM rewards and the heaviest group wins.

    Args:
        n: Number of candidates to sample.

    Returns:
        The best-scored candidate of the heaviest behaviour group.
    """
    # Score each candidate with the PRM (weakest step) and group candidates by behaviour
    candidates = []
    for i in range(n):
        answer = sample()
        reward, tests = min(prm_scores(split_steps(answer)) or [0.0]), verify(answer)
        candidates.append((answer, reward, tests, behaviour(answer)))
        print(f"candidate {i + 1}: PRM {reward:.2f}  unit tests {tests:.2f}")

    # Weighted vote: each answer (behaviour) collects the PRM rewards of every candidate that gives it
    weights, members = defaultdict(float), defaultdict(list)
    for candidate in candidates:
        weights[candidate[3]] += candidate[1]
        members[candidate[3]].append(candidate)

    print("\nAnswers by total PRM weight (outputs on the test inputs):")
    for key in sorted(weights, key=weights.get, reverse=True):
        print(f"  weight {weights[key]:.2f} from {len(members[key])} candidate(s), unit tests {members[key][0][2]:.2f}: {key}")

    # The heaviest answer wins; its best-scored candidate is the code we return
    best_key = max(weights, key=weights.get)
    best_answer, best_reward, best_tests, _ = max(members[best_key], key=lambda c: c[1])
    plain_pick = max(candidates, key=lambda c: c[1])
    print(f"\nWeighted pick: weight {weights[best_key]:.2f}, unit tests {best_tests:.2f}"
          f" (plain Best-of-N pick: unit tests {plain_pick[2]:.2f}; best possible {max(c[2] for c in candidates):.2f})\n{best_answer}")
    return best_answer


if __name__ == "__main__":
    # Needs the Docker stack with qwen3.5-4b loaded
    best_of_n_weighted_prm()
