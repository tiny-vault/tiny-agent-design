"""Best-of-N with an outcome reward model (ORM): sample 10 ``roman_to_int`` candidates and keep the one the judge rates highest.

**What it is.** Best-of-N draws N answers from a model and keeps the best one according to a verifier. An outcome reward model rates each finished answer as a whole, without looking at how it was built; the highest rating wins.

**How it works here.** ``best_of_n_orm(n)`` samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1`` so candidates differ). ``orm_score`` asks ``judge`` (the same model at ``temperature=0``, standing in for a trained reward model) to rate each answer 0 to 10 with ``ORM_PROMPT``. Its score is the model's opinion, not a measured quality, so ``verify`` also runs the unit tests afterwards, only to show whether the ORM's pick really is the best.

**Running it.** ``python orm.py``. Scores vary between runs. Compare ``prm``, which scores every step instead of the whole answer, and ``compare`` for all five verifiers on the same candidates.

Warning:
    ``verify`` runs model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

import re
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

# Ground truth for verify; the reward model never sees it
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

    Here it only reports how good the ORM's pick was; it does not choose.

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


def sample():
    """Draw one candidate: a fresh reply to ``query`` from the temperature-1 ``llm``."""
    return llm.generate([{"role": "user", "content": query}]).content


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


if __name__ == "__main__":
    # Needs the Docker stack with qwen3.5-4b loaded
    best_of_n_orm()
