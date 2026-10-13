"""Zero-shot prompting: ask the question with no solved example, plain and with "Let's think step by step."

**What it is.** The number of solved examples placed before a question is the "shot" count; zero-shot has none, so the model sees only the question. Chain-of-thought (CoT) for zero-shot is the trigger phrase "Let's think step by step." (Kojima et al., 2022), which makes the model reason before it answers.

**How it works here.** ``zero_shot(cot)`` sends the penguin ``query`` (answer 11) alone, or with the trigger phrase when ``cot`` is True. ``show`` prints the exact messages sent, the reply and a result line with the answer (``final_number`` takes the last number of the reply), whether it is correct and the tokens. Plain zero-shot has no example to copy, so the reply style is up to the model.

**Running it.** ``python zero_shot.py`` runs the plain variant, then the CoT variant. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``, as in the book, and each variant runs once, so one run proves nothing about which is better. Compare ``one_shot`` and ``few_shot`` for the same question with examples, and ``compare`` for all six variants in one table.
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
    print(f"SENT ({len(messages)} message)")
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


def zero_shot(cot=False):
    """Zero-shot: no solved examples; with ``cot`` the trigger "Let's think step by step." is added.

    Args:
        cot: If True, append the trigger phrase so the model reasons before answering.

    Returns:
        The model's ``Response``.
    """
    # With no example to imitate, zero-shot CoT needs the trigger phrase to start reasoning
    messages = [{"role": "user", "content": query + (" Let's think step by step." if cot else "")}]
    response = llm.generate(messages)
    show(f"zero_shot{' + CoT' if cot else ''}", messages, response)
    return response


if __name__ == "__main__":
    # Plain, then CoT (needs the Docker stack with qwen3.5-4b loaded)
    for cot in (False, True):
        zero_shot(cot=cot)
