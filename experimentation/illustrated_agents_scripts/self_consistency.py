"""Self-consistency: ask the same chain-of-thought question 10 times and take a majority vote.

Technique:
    Self-consistency (Wang et al., 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models") samples several reasoning paths for one question and keeps the most frequent final answer. No reward model or verifier is involved; agreement between samples does the selecting. It relies on sampling (here ``temperature=1``), so the paths differ; with greedy decoding every run would give the same answer and the vote would be pointless. Each reply ends with ``Answer:``, so the final answer can be cut out of the text and counted with :class:`collections.Counter`.

Messages sent:
    1. ``user``: the seating puzzle (:data:`query`), sent 10 separate times as fresh one-message chats.

Expected output:
    A ``Counter`` of the 10 extracted answers. The correct answer is seat 5 (Chris 1, Jor 2, Sarah 3, Ilse 4, Maarten 5, Irene 6), so the most common entry should be some form of ``5``. Small models sometimes write it differently (``5``, ``Seat 5``, ``**5**``), which splits the vote across spellings. Ten calls take a few minutes on the local GPU.

Run:
    ``.venv/Scripts/python trigger_ias.py self_consistency`` from the repo root.

Requires:
    The Docker stack up with ``qwen3.5-4b`` loaded (``node execute.mjs up``, ``node execute.mjs load qwen3.5-4b``) and the secret file ``experimentation/chat-app/secrets/llama_api_key``, which ``up`` creates. Importing the module reads the secret but makes no network call.
"""

from pathlib import Path
from collections import Counter

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)
"""The model client, ``qwen3.5-4b`` through the chat-app gateway at ``http://localhost:3200/v1``.

``temperature=1`` matters here: self-consistency needs varied reasoning paths to vote over. Building the client makes no network call; ``LLM.generate`` does.

:meta hide-value:
"""

query = """Six friends (Maarten, Ilse, Sarah, Jor, Irene, and Chris)
    are sitting in a row.

- Sarah is in seat 3.
- Chris is sitting at one of the ends of the row.
- Jor is sitting immediately to the right of Chris.
- Ilse is sitting exactly in the middle of Sarah and Maarten.
- Irene is not sitting next to Sarah.
- Maarten is sitting somewhere to the left of Irene.

In which seat is Maarten sitting?
Let's think step by step and give back your answer only after 'Answer:'.
"""
"""The seating puzzle, verbatim from the book; the correct answer is seat 5."""

if __name__ == "__main__":
    # Sample 10 reasoning paths (needs the Docker stack with qwen3.5-4b loaded)
    answers = []
    for i in range(10):
        messages = [{"role": "user", "content": query}]
        response = llm.generate(messages)
        # Keep only the text after the last 'Answer:', so the vote counts final answers, not reasoning
        answer = response.content.split("Answer:")[-1].replace("\n", "").strip()
        answers.append(answer)
        print(f"sample {i + 1}: {answer}")

    # Majority vote: the most common answer wins
    votes = Counter(answers)
    print(votes)
    print(f"majority answer: {votes.most_common(1)[0][0]}")
