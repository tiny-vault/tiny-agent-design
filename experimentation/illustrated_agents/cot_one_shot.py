"""One-shot chain-of-thought: the solved example shows its working, so the model does too.

The chat is the flamingo example (``prompt_example``), its step-by-step answer (``prompt_answer``, 6 - 2 = 4, 4 - 1 = 3), then the penguin ``query`` (answer 11). Expect 9 - 2 = 7, 7 + 4 = 11 in about 100 to 115 tokens. ``show`` prints each ``Response`` field on its own line. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``. Compare ``one_shot`` and ``zero_shot_cot``.
"""

# qwen3.5-4b on the Docker stack (key from the Docker secret), the penguin query and show
from common import flamingo_cot_answer, flamingo_example, llm, penguin_query as query, show

# Example of one-shot chain-of-thought prompting
prompt_example = flamingo_example

prompt_answer = flamingo_cot_answer

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
