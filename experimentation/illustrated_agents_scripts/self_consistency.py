"""Self-consistency (Wang et al., 2022): sample 10 chain-of-thought answers and take a majority vote.

``query`` is the book's seating puzzle (answer: seat 5); the text after the last ``Answer:`` is counted with ``Counter``. ``llm`` is ``qwen3.5-4b`` via the gateway; ``temperature=1`` varies the reasoning paths. Small models may split the vote across spellings (``5``, ``Seat 5``, ``**5**``).
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
