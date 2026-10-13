"""Best-of-N five ways: generate 10 ``roman_to_int`` candidates and pick one with unit tests, an ORM, a PRM, a weighted PRM vote or a plain majority vote.

**What it is.** Best-of-N draws N answers from a model and keeps the best one according to a verifier. The model never improves; the selection does. Everything depends on how good the verifier is.

**The five ways to pick.**

- ``best_of_n_majority_vote``: self-consistency (Wang et al., 2022), the base variant. No verifier at all: candidates that return the same outputs on the test inputs (``behaviour``) vote for one answer, and the answer with the most votes wins. Agreement is evidence of correctness, not a guarantee.
- ``best_of_n_unit_tests``: runs the candidate on ``test_cases`` (``verify``, a fraction 0.0 to 1.0; a crash or Markdown-fenced reply scores 0.0). Real ground truth.
- ``best_of_n_orm``: an outcome reward model rates each finished answer 0 to 10 (``orm_score`` with ``ORM_PROMPT``).
- ``best_of_n_prm``: a process reward model rates every non-empty code line (``split_steps``, ``prm_scores`` with ``PRM_PROMPT``) and the candidate with the best weakest step wins (``min``).
- ``best_of_n_weighted_prm``: groups candidates that return the same outputs on the test inputs (``behaviour``), sums their PRM rewards and keeps the heaviest group, so an answer many good samples agree on beats one lucky outlier. It is the majority vote with each vote weighted by a reward.

**How it works here.** Every method samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1`` so candidates differ) and differs only in how it picks. No trained reward model is in the stack, so ``judge`` (the same model at ``temperature=0``) stands in for both; its scores are the model's opinion, not a measured quality. The reward and voting methods still run ``verify`` afterwards to show whether their pick really is the best. The majority vote compares behaviour on the test inputs, not the code text, so differently written but equivalent solutions count as one answer; a model that is consistently wrong still wins the vote.

**Running it.** ``python best_of_n.py orm`` runs one method (``majority_vote``, ``unit_tests``, ``orm``, ``prm``, ``weighted_prm``); no argument runs all five. Scores vary between runs, so one run does not rank the verifiers. Compare ``shot_prompting``, which tries single prompts instead of sampling many.

Warning:
    ``verify`` and ``behaviour`` run model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
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

    This is the ground-truth verifier of Best-of-N: unlike the reward models it checks real behaviour.

    Args:
        answer: The model's reply, expected to be plain Python defining ``roman_to_int``.

    Returns:
        A score from 0.0 to 1.0; 0.0 if the code does not run, lacks ``roman_to_int`` or crashes.

    Warning:
        Runs model-written code with ``exec`` and no sandbox.
    """
    try:
        # Extract the function
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]

        # Evaluate the function on test cases
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


def best_of_n_majority_vote(n=10):
    """Self-consistency: candidates with identical behaviour vote, and the most common answer wins.

    Args:
        n: Number of candidates to sample.

    Returns:
        The first candidate of the answer (behaviour group) with the most votes.
    """
    # No verifier: group candidates by what they output on the test inputs; one candidate, one vote
    candidates = []
    for i in range(n):
        answer = sample()
        tests = verify(answer)
        candidates.append((answer, tests, behaviour(answer)))
        print(f"candidate {i + 1}: unit tests {tests:.2f}")

    members = defaultdict(list)
    for candidate in candidates:
        members[candidate[2]].append(candidate)

    print("\nAnswers by number of votes (outputs on the test inputs):")
    for key in sorted(members, key=lambda k: len(members[k]), reverse=True):
        print(f"  {len(members[key])} vote(s), unit tests {members[key][0][1]:.2f}: {key}")

    # The answer with the most votes wins; a tie goes to the group seen first
    best_key = max(members, key=lambda k: len(members[k]))
    best_answer, best_tests, _ = members[best_key][0]
    print(f"\nMajority pick: {len(members[best_key])} of {n} votes, unit tests {best_tests:.2f}"
          f" (best possible {max(c[1] for c in candidates):.2f})\n{best_answer}")
    return best_answer


def best_of_n_unit_tests(n=10):
    """Best-of-N where the unit tests are the verifier.

    Args:
        n: Number of candidates to sample.

    Returns:
        The candidate with the highest ``verify`` score (the first one if several tie).
    """
    # The verifier is the unit tests themselves: the most tests passed wins
    responses = []
    for i in range(n):
        answer = sample()
        score = verify(answer)
        responses.append((answer, score))
        print(f"candidate {i + 1}: score {score:.2f}")

    best_answer, best_score = max(responses, key=lambda x: x[1])
    print([response[1] for response in responses])
    print(f"\nbest score: {best_score:.2f}\n{best_answer}")
    return best_answer


def best_of_n_orm(n=10):
    """Best-of-N where an outcome reward model picks; ``verify`` only reports how good the pick was.

    Args:
        n: Number of candidates to sample.

    Returns:
        The candidate with the highest ORM score.
    """
    # The ORM scores each finished answer as a whole; the top score wins
    candidates = []
    for i in range(n):
        answer = sample()
        reward, tests = orm_score(answer), verify(answer)
        candidates.append((answer, reward, tests))
        print(f"candidate {i + 1}: ORM {reward:.2f}  unit tests {tests:.2f}")

    # The unit tests show whether the ORM's pick was the right one
    best_answer, best_reward, best_tests = max(candidates, key=lambda c: c[1])
    print(f"\nORM pick: ORM {best_reward:.2f}, unit tests {best_tests:.2f} (best possible {max(c[2] for c in candidates):.2f})\n{best_answer}")
    return best_answer


def best_of_n_prm(n=10):
    """Best-of-N where a process reward model picks by each candidate's weakest step.

    Args:
        n: Number of candidates to sample.

    Returns:
        The candidate whose lowest step score is highest. Also prints that pick's step scores.
    """
    # The PRM scores every step; a solution is only as good as its weakest step (min; the product is another choice)
    candidates = []
    for i in range(n):
        answer = sample()
        steps = split_steps(answer)
        scores = prm_scores(steps) or [0.0]
        reward, tests = min(scores), verify(answer)
        weakest = scores.index(reward)
        candidates.append((answer, reward, tests, list(zip(steps, scores))))
        print(f"candidate {i + 1}: PRM {reward:.2f} (weakest step {weakest + 1}/{len(steps)}: {steps[weakest].strip() if steps else '-'})  unit tests {tests:.2f}")

    best_answer, best_reward, best_tests, best_steps = max(candidates, key=lambda c: c[1])
    print(f"\nPRM pick: PRM {best_reward:.2f}, unit tests {best_tests:.2f} (best possible {max(c[2] for c in candidates):.2f})\n{best_answer}")
    print("\nStep scores of the pick:")
    for step, score in best_steps:
        print(f"  {score:.2f}  {step}")
    return best_answer


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


methods = {
    "majority_vote": best_of_n_majority_vote,
    "unit_tests": best_of_n_unit_tests,
    "orm": best_of_n_orm,
    "prm": best_of_n_prm,
    "weighted_prm": best_of_n_weighted_prm,
}

if __name__ == "__main__":
    # One method by name, or all five (needs the Docker stack with qwen3.5-4b loaded).
    # Via trigger_ias.py, argv[1] is the script name, which matches no method, so all five run.
    chosen = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] in methods else list(methods)
    for name in chosen:
        print(f"\n===== best_of_n_{name} =====")
        methods[name]()
