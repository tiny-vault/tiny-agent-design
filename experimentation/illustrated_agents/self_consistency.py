"""Self-consistency (Wang et al., 2022): sample 10 chain-of-thought answers and take a majority vote.

``query`` is the book's seating puzzle (answer: seat 5); the text after the last ``Answer:`` is counted with ``Counter``. ``llm`` is ``qwen3.5-4b`` via the gateway; ``temperature=1`` varies the reasoning paths. Small models may split the vote across spellings (``5``, ``Seat 5``, ``**5**``).
"""

from collections import Counter

# qwen3.5-4b on the Docker stack, temperature=1 (key from the Docker secret)
from common import llm, seating_query as query

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
