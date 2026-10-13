"""Zero-, one- and few-shot prompting, each plain and with chain-of-thought, compared on the same question.

**What it is.** The number of solved examples placed before a question is the "shot" count: zero-shot has none, one-shot has one, few-shot has several. Chain-of-thought (CoT) adds step-by-step working, either in the examples' answers or, for zero-shot, with the phrase "Let's think step by step." (Kojima et al., 2022).

**One file, not four.** This script replaces the book's separate one-shot, chain-of-thought one-shot and zero-shot chain-of-thought cells, which are ``one_shot(cot=False)``, ``one_shot(cot=True)`` and ``zero_shot(cot=True)`` here; zero-shot without CoT and few-shot are new.

**Why compare them.** Each choice trades prompt length against reply style: examples cost prompt tokens, CoT costs completion tokens.

**How it works here.** Every method asks the penguin ``query`` (answer 11) and differs only in what comes before it. ``zero_shot`` sends the query alone; ``one_shot`` first shows the flamingo example (``examples[0]``); ``few_shot`` shows all three ``examples`` (flamingos, apples, birds). With ``cot=False`` each example's answer is the bare number, so the model copies that format (expect ``11`` in a few tokens). With ``cot=True`` the answers show their working, so the model reasons first (about 50 to 150 tokens).

**Reading the output.** ``build_messages`` turns ``k`` examples into the chat; ``final_number`` takes the last number of a reply; ``show`` prints one run as a labelled block (what this run changes, the exact messages sent, the reply, and a result line with the answer, whether it is correct and the tokens). ``compare`` runs all six and ends with a table. ``python shot_prompting.py few_shot`` runs one method (plain and CoT); no argument runs ``compare``.

**Limits.** ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``, as in the book, and each variant runs once, so the table is illustrative: it cannot show that one method is better. Compare ``best_of_n``, whose ``best_of_n_majority_vote`` samples many answers and votes.
"""

import re
import sys
import textwrap
from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

# The query we want to ask the model, and its answer (9 - 2 + 4)
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"
expected = 11

# Solved examples: (question, bare answer, answer that shows its working). The first is the book's flamingo example.
examples = [
    (
        "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?",
        "3",
        "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3.",
    ),
    (
        "I had 5 apples. I ate 1 and my friend gave me 3 more. How many apples do I have?",
        "7",
        "You had 5 apples and ate 1. 5 - 1 = 4. Then your friend gave you 3. You have now 4 + 3 = 7.",
    ),
    (
        "There were 8 birds on a wire. 3 flew off, then 2 landed and 1 more flew off. How many birds are on the wire?",
        "6",
        "There were 8 birds and 3 flew off. 8 - 3 = 5. Then 2 landed. 5 + 2 = 7. Then 1 flew off. 7 - 1 = 6.",
    ),
]


def describe(k, cot):
    """Return one line saying how a run differs: its number of solved examples and the answer style they teach."""
    # One line saying what this run changes, so the six blocks are told apart at a glance
    shots = "no solved example" if k == 0 else f"{k} solved example{'s' if k > 1 else ''} before the question"
    if not cot:
        style = "answers are bare numbers, so expect a bare number back"
    elif k == 0:
        style = "adds \"Let's think step by step.\", so expect reasoning first"
    else:
        style = "answers show their working, so expect reasoning first"
    return f"{shots}; {style}"


def indent(text, prefix="    "):
    """Indent a reply for printing; an empty reply prints as ``(empty)``."""
    return textwrap.indent((text or "(empty)").strip(), prefix)


def show(label, k, cot, messages, response):
    """Print one run as a block: what changed, the exact messages sent, the reply and a correct/wrong line.

    Args:
        label: Heading for the block.
        k: Number of solved examples in the prompt.
        cot: Whether the examples (or the trigger phrase) elicit step-by-step working.
        messages: The chat that was sent.
        response: The ``Response`` the model returned.
    """
    # One run as a block: what changed, what was sent, what came back, and whether it was right
    answer = final_number(response.content)
    verdict = "CORRECT" if answer == expected else "WRONG"
    print("=" * 78)
    print(f"{label}")
    print(f"  {describe(k, cot)}")
    print("-" * 78)
    print(f"SENT ({len(messages)} messages; the last one is the real question)")
    for i, message in enumerate(messages, 1):
        print(f"  {i}. {message['role']:<9} {message['content']}")
    if response.reasoning:
        print("REASONING (the model's hidden thinking)")
        print(indent(response.reasoning))
    print("REPLY")
    print(indent(response.content))
    if response.tool_call:
        print(f"TOOL CALL  {response.tool_call}")
    print(f"RESULT  answer {answer} (expected {expected}) -> {verdict}   "
          f"tokens: {response.metadata['prompt_tokens']} prompt, {response.metadata['completion_tokens']} completion")
    print()


def final_number(text):
    """Return the last integer in a reply, or None if there is none.

    A heuristic: it would misread a reply that mentions another number after its answer.
    """
    # The answer is the last number in the reply, whether bare ("11") or after the working ("... = 11.")
    numbers = re.findall(r"-?\d+", text or "")
    return int(numbers[-1]) if numbers else None


def build_messages(k, cot):
    """Build the chat for a run: k solved examples as user/assistant turns, then the penguin question.

    Args:
        k: How many of ``examples`` to show (0 is zero-shot, 1 one-shot, 3 few-shot).
        cot: If True the examples answer with their working; if False with a bare number.

    Returns:
        A list of chat messages ending with the real question.
    """
    # k solved examples as user/assistant turns, then the real question
    messages = []
    for question, bare, working in examples[:k]:
        # Plain examples end with "Be concise." so the bare number reads as the requested format
        messages.append({"role": "user", "content": question if cot else question + " Be concise."})
        messages.append({"role": "assistant", "content": working if cot else bare})
    # With no example to imitate, zero-shot CoT needs the trigger phrase to start reasoning
    messages.append({"role": "user", "content": query + (" Let's think step by step." if cot and k == 0 else "")})
    return messages


def run(label, k, cot):
    """Send one prompt variant to the model, print it with ``show`` and return the ``Response``."""
    messages = build_messages(k, cot)
    response = llm.generate(messages)
    show(label, k, cot, messages, response)
    return response


def zero_shot(cot=False, label="zero_shot"):
    """Zero-shot: no solved examples; with ``cot`` the trigger "Let's think step by step." is added."""
    return run(label, 0, cot)


def one_shot(cot=False, label="one_shot"):
    """One-shot: one solved example (the flamingos) before the question."""
    return run(label, 1, cot)


def few_shot(cot=False, label="few_shot"):
    """Few-shot: all three solved examples before the question."""
    return run(label, len(examples), cot)


methods = {
    "zero_shot": zero_shot,
    "one_shot": one_shot,
    "few_shot": few_shot,
}


def compare():
    """Run all six variants (three methods, plain and CoT) and print a table of answer, correctness and tokens.

    Returns:
        One ``(name, answer, metadata)`` row per run. One sample each at temperature 1, so the table is
        illustrative and says nothing reliable about which method is better.
    """
    # Run each method plain and with CoT, keep its reply, then print one table row per run
    rows = []
    runs = [(name, method, cot) for name, method in methods.items() for cot in (False, True)]
    for i, (name, method, cot) in enumerate(runs, 1):
        label = f"RUN {i}/{len(runs)}: {name}{' + CoT' if cot else ''}"
        response = method(cot=cot, label=label)
        rows.append((f"{name}{' + CoT' if cot else ''}", final_number(response.content), response.metadata))

    # More examples cost more prompt tokens; CoT costs more completion tokens
    print("=" * 78)
    print(f"SUMMARY  question: {query}")
    print(f"         expected answer: {expected}")
    print("-" * 78)
    print(f"{'run':<5}{'method':<18}{'answer':>8}{'correct':>9}{'prompt tok':>12}{'completion tok':>16}")
    for i, (label, answer, metadata) in enumerate(rows, 1):
        correct = "yes" if answer == expected else "NO"
        print(f"{i:<5}{label:<18}{str(answer):>8}{correct:>9}{metadata['prompt_tokens']:>12}{metadata['completion_tokens']:>16}")
    return rows


if __name__ == "__main__":
    # One method by name (plain, then CoT), or the full comparison (needs the Docker stack with qwen3.5-4b loaded).
    # Via trigger_ias.py, argv[1] is the script name, which matches no method, so compare runs.
    if len(sys.argv) > 1 and sys.argv[1] in methods:
        for i, cot in enumerate((False, True), 1):
            methods[sys.argv[1]](cot=cot, label=f"RUN {i}/2: {sys.argv[1]}{' + CoT' if cot else ''}")
    else:
        compare()
