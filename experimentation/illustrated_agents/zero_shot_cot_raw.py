"""Same as ``zero_shot_cot``, but prints the raw ``Response`` repr, as the book's notebook does.

Expect one line: ``Response(content='...11...', reasoning=None, tool_call=None, metadata={...})``. ``llm`` is ``qwen3.5-4b`` via the gateway at ``temperature=1``.
"""

# qwen3.5-4b on the Docker stack (key from the Docker secret) and the penguin query
from common import llm, penguin_query as query, step_by_step

# Messages with zero-shot Chain-of-Thought prompting
messages = [{"role": "user", "content": query + " " + step_by_step}]

if __name__ == "__main__":
    # Generate response (needs the Docker stack with qwen3.5-4b loaded)
    # Print the whole Response object (all fields, as its repr)
    print(llm.generate(messages))
