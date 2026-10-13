"""Best-of-N by majority vote (self-consistency): sample 10 ``roman_to_int`` candidates and keep the answer most of them agree on.

**What it is.** Self-consistency (Wang et al., 2022) is the base Best-of-N variant. There is no verifier at all: candidates that return the same outputs on the test inputs (``behaviour``) vote for one answer, and the answer with the most votes wins. Agreement is evidence of correctness, not a guarantee; a model that is consistently wrong still wins the vote.

**How it works here.** ``best_of_n_majority_vote(n)`` samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1`` so candidates differ). ``behaviour`` fingerprints each candidate by its outputs on the test inputs, not its code text, so differently written but equivalent solutions count as one answer. ``verify`` runs the unit tests afterwards, only to show whether the winning answer is really correct.

**Running it.** ``python majority_vote.py``. Results vary between runs. Compare ``weighted_prm``, the same vote with each vote weighted by a reward, and ``compare`` for all five verifiers on the same candidates.

Warning:
    ``verify`` and ``behaviour`` run model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

from collections import defaultdict
from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[2] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

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

# Ground truth for verify; behaviour uses only the inputs
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

    Here it only reports how good the vote's pick was; it does not choose.

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


if __name__ == "__main__":
    # Needs the Docker stack with qwen3.5-4b loaded
    best_of_n_majority_vote()
