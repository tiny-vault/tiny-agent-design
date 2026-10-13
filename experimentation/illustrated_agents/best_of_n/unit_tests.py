"""Best-of-N with unit tests: sample 10 ``roman_to_int`` candidates and keep the one that passes the most tests.

**What it is.** Best-of-N draws N answers from a model and keeps the best one according to a verifier. The model never improves; the selection does. Here the verifier is real ground truth: ``verify`` runs each candidate on ``test_cases`` and scores the fraction it passes (0.0 to 1.0; a crash or Markdown-fenced reply scores 0.0).

**How it works here.** ``best_of_n_unit_tests(n)`` samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1`` so candidates differ), scores each with ``verify`` and returns the highest. This is the strongest verifier, but it needs tests, which most real tasks do not have.

**Running it.** ``python unit_tests.py``. Scores vary between runs. Compare ``orm``, ``prm``, ``majority_vote`` and ``weighted_prm`` for verifiers that need no tests, and ``compare`` for all five on the same candidates.

Warning:
    ``verify`` runs model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

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

# Ground truth for verify
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

    This is the ground-truth verifier of Best-of-N: it checks real behaviour.

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


def sample():
    """Draw one candidate: a fresh reply to ``query`` from the temperature-1 ``llm``."""
    return llm.generate([{"role": "user", "content": query}]).content


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


if __name__ == "__main__":
    # Needs the Docker stack with qwen3.5-4b loaded
    best_of_n_unit_tests()
