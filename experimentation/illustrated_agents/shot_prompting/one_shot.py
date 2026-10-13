"""One-shot prompting: show one solved example (the flamingos) before the question, with a bare answer or with its working.

**What it is.** One solved example placed before the question is "one-shot". The model imitates the example's answer style: a bare number invites a bare number back, while an example that shows its working (chain-of-thought, CoT) makes the model reason first.

**How it works here.** ``one_shot(cot)`` sends the flamingo ``example`` as a user/assistant pair, then the penguin ``query`` (answer 11). With ``cot=False`` the example's answer is the bare ``3`` (expect ``11`` in a few tokens); with ``cot=True`` it shows its working (expect about 50 to 150 tokens of reasoning). ``show`` prints the exact messages sent, the reply and a result line with the answer (``final_number`` takes the last number of the reply), whether it is correct and the tokens.

**Running it.** ``python one_shot.py`` runs the plain variant, then the CoT variant. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``, as in the book, and each variant runs once, so one run proves nothing about which is better. Compare ``zero_shot`` (no example) and ``few_shot`` (three examples), and ``compare`` for all six variants in one table.
"""

import re
import textwrap
from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[2] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

# The query we want to ask the model, and its answer (9 - 2 + 4)
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"
expected = 11

# The solved example (the book's flamingo one): (question, bare answer, answer that shows its working)
example = (
    "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?",
    "3",
    "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3.",
)


def indent(text, prefix="    "):
    """Indent a reply for printing; an empty reply prints as ``(empty)``."""
    return textwrap.indent((text or "(empty)").strip(), prefix)


def final_number(text):
    """Return the last integer in a reply, or None if there is none.

    A heuristic: it would misread a reply that mentions another number after its answer.
    """
    # The answer is the last number in the reply, whether bare ("11") or after the working ("... = 11.")
    numbers = re.findall(r"-?\d+", text or "")
    return int(numbers[-1]) if numbers else None


def show(label, messages, response):
    """Print one run as a block: the exact messages sent, the reply and a correct/wrong line.

    Args:
        label: Heading for the block.
        messages: The chat that was sent.
        response: The ``Response`` the model returned.
    """
    answer = final_number(response.content)
    verdict = "CORRECT" if answer == expected else "WRONG"
    print("=" * 78)
    print(label)
    print("-" * 78)
    print(f"SENT ({len(messages)} messages; the last one is the real question)")
    for i, message in enumerate(messages, 1):
        print(f"  {i}. {message['role']:<9} {message['content']}")
    if response.reasoning:
        print("REASONING (the model's hidden thinking)")
        print(indent(response.reasoning))
    print("REPLY")
    print(indent(response.content))
    print(f"RESULT  answer {answer} (expected {expected}) -> {verdict}   "
          f"tokens: {response.metadata['prompt_tokens']} prompt, {response.metadata['completion_tokens']} completion")
    print()


def one_shot(cot=False):
    """One-shot: the flamingo example, then the penguin question.

    Args:
        cot: If True the example answers with its working; if False with a bare number.

    Returns:
        The model's ``Response``.
    """
    question, bare, working = example
    # Plain examples end with "Be concise." so the bare number reads as the requested format
    messages = [
        {"role": "user", "content": question if cot else question + " Be concise."},
        {"role": "assistant", "content": working if cot else bare},
        {"role": "user", "content": query},
    ]
    response = llm.generate(messages)
    show(f"one_shot{' + CoT' if cot else ''}", messages, response)
    return response


if __name__ == "__main__":
    # Plain, then CoT (needs the Docker stack with qwen3.5-4b loaded)
    for cot in (False, True):
        one_shot(cot=cot)
