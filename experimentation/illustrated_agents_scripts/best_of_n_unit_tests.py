"""Best-of-N with a verifier: generate 10 ``roman_to_int`` functions and keep the one passing most unit tests.

``query`` is the book's coding task. ``verify`` returns the fraction of ``test_cases`` passed (the last six are invalid inputs that must return 0); a crash or a Markdown-fenced reply scores 0.0. ``llm`` is ``qwen3.5-4b`` via the gateway; ``temperature=1`` makes the candidates differ. Compare ``self_consistency``, which votes without a verifier.

Warning:
    ``verify`` runs model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

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
    try:
        # Extract the function
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]

        # Evaluate the function on test cases
        # (the book's listing wraps `len(test_cases)` across two lines; as printed, `/ len` divides by the
        # builtin, raises TypeError and every candidate scores 0.0, so it is joined here)
        score = sum(fn(inp) == expected for inp, expected in test_cases) / len(test_cases)
        return score
    except Exception:
        # Return 0 if the code is not valid Python
        return 0.0


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

if __name__ == "__main__":
    # Generate 10 candidates (needs the Docker stack with qwen3.5-4b loaded)
    responses = []
    for i in range(10):
        response = llm.generate([{"role": "user", "content": query}])

        # Verify the response and compute a score
        score = verify(response.content)

        # Store the response and its score
        responses.append((response.content, score))
        print(f"candidate {i + 1}: score {score:.2f}")

    # Get highest score and the corresponding answer
    best_answer, best_score = max(responses, key=lambda x: x[1])

    # Show all scores, then the winner
    print([response[1] for response in responses])
    print(f"\nbest score: {best_score:.2f}\n{best_answer}")
