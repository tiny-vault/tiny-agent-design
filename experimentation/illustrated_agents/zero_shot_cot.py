"""Zero-shot chain-of-thought: no example, just "Let's think step by step." (Kojima et al., 2022).

One ``user`` message: the penguin ``query`` (answer 11) plus the trigger. Shortest prompt, longest reply: a numbered breakdown ending in 11. ``show`` prints each ``Response`` field on its own line. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``. ``zero_shot_cot_raw`` prints the raw ``Response`` instead.
"""

# qwen3.5-4b on the Docker stack (key from the Docker secret), the penguin query and show
from common import llm, penguin_query as query, show, step_by_step

# Messages with zero-shot Chain-of-Thought prompting
messages = [{"role": "user", "content": query + " " + step_by_step}]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    show(llm.generate(messages))
