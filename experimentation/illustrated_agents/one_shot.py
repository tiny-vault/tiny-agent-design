"""One-shot prompting: one solved example, then the real question.

The chat is the flamingo example (``prompt_example``, ending "Be concise."), its bare answer ``"3"`` (``prompt_answer``), then the penguin ``query`` (answer 11). The model copies the bare-number format: expect ``11`` in about 3 tokens. ``show`` prints each ``Response`` field on its own line. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``, as in the book. Compare ``cot_one_shot``.
"""

# qwen3.5-4b on the Docker stack (key from the Docker secret), the penguin query and show
from common import flamingo_answer, flamingo_example, llm, penguin_query as query, show

# Example of one-shot prompting
prompt_example = flamingo_example + " Be concise."

prompt_answer = flamingo_answer

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
