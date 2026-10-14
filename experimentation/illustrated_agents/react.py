"""ReAct (Reason and Act): the prompt that asks for one THOUGHT and one ACTION per turn, and the parser that splits them.

**What it shows.** A plain model call can answer or call one tool, but a task such as "add, then multiply, then subtract" needs several turns. ReAct puts the cycle in the prompt: each turn the model writes ``THOUGHT:`` (its reasoning) and ``ACTION:`` (a JSON tool call), the harness runs the action and answers with an ``OBSERVATION:``, and the model goes again until it calls the ``final_answer`` tool.

**How it works here.** ``react`` is the book's ``ReAct`` with ``max_steps=10``. ``react.prompt`` is the system message and ``react.parse`` turns a reply into a ``Response`` whose ``reasoning`` is the THOUGHT and whose ``content`` is only the ACTION. ``sample_reply`` is parsed first, with no model, to show the split; then ``query`` is sent to ``llm`` once under the ReAct prompt and its reply is parsed the same way. One call is one step: the loop that feeds back the observation lives in ``TinyAgent.run``.

**What to expect.** The parsed sample is deterministic. The model's reply varies, but should start with ``THOUGHT:`` and hold one ``add`` action. ``llm`` is ``qwen3.5-4b`` via the gateway with ``temperature=None``. Compare ``agent_with_memory`` and the repo's ``ReAct`` in ``tiny_agent/components/planning/react.py``, which also drives the loop.
"""

from pathlib import Path

from illustrated_agents.chapters.ch2 import LLM
from illustrated_agents.chapters.ch6 import ReAct

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

react = ReAct(max_steps=10)

# The book's multi-step task
query = "What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?"

# A reply in the format the prompt asks for, including an observation the model must not invent
sample_reply = """THOUGHT: I should add the two numbers first.
ACTION:
{"tool": "add", "kwargs": {"a": "4.6", "b": "6.685"}}
OBSERVATION: 11.285"""

# Tool descriptions for the model; nothing here runs them
tools_prompt = "# Tools\n`add`: add(a: str, b: str)\n`subtract`: subtract(a: str, b: str)\n`multiply`: multiply(a: str, b: str)\n"


def show(label, response):
    """Print the THOUGHT and the ACTION a parsed ``Response`` holds."""
    print(f"{label}\n  thought: {response.reasoning!r}\n  action:  {response.content!r}\n")


if __name__ == "__main__":
    # No model needed: the OBSERVATION line is dropped by the parser
    from illustrated_agents.chapters.ch2 import Response

    show("parsed sample", react.parse(Response(content=sample_reply)))

    # One step from the model (needs the Docker stack with qwen3.5-4b loaded)
    messages = [
        {"role": "system", "content": "You are a helpful assistant.\n\n" + react.prompt + tools_prompt},
        {"role": "user", "content": query},
    ]
    show("model step", react.parse(llm.generate(messages)))
