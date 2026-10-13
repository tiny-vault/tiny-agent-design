"""Self-consistency (Wang et al., 2022): sample 10 chain-of-thought answers and take a majority vote.

**What it is.** Ask the same reasoning question many times with randomness on, then keep the answer most samples agree on. Different reasoning paths can reach the same final answer, so agreement is evidence that it is right. It is not a guarantee.

**Why it exists.** A single chain-of-thought sample can go wrong at any step. Voting across samples reduces the effect of one bad path, at the cost of N model calls.

**How it works here.** ``query`` is the book's seating puzzle (answer: seat 5), asking for the answer only after "Answer:". Ten times the script sends it to ``llm`` (``qwen3.5-4b`` via the gateway; ``temperature=1`` so the reasoning paths differ) and keeps only the text after the last "Answer:", so the vote counts final answers and not the reasoning. ``Counter`` tallies them and the most common wins.

**What to expect.** Ten lines ``sample i: <answer>``, the tally, then the majority answer. Small models may split the vote across spellings of one answer (``5``, ``Seat 5``, ``**5**``), because the votes compare raw strings. Output varies between runs.

Compare ``best_of_n``, which also samples many candidates but picks with a verifier.
"""

from pathlib import Path
from collections import Counter

from illustrated_agents.chapters.ch2 import LLM

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key, temperature=1)

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
