"""Native ReAct on ``tiny_agent``: the think, act, observe loop with the model's own tool calling instead of a THOUGHT / ACTION text format.

**What it shows.** ``ReAct`` teaches the cycle in the system prompt and parses ``THOUGHT:`` and ``ACTION:`` out of the reply text. A model trained for tool calling needs none of that: the tool schemas travel in the request's ``tools`` field and the call comes back as a structured ``tool_call``. ``NativeReAct`` is therefore ``ReAct`` with an empty ``prompt`` and a ``parse`` that returns the reply unchanged; only the loop in ``TinyAgent.run`` remains.

**How it works here.** Everything comes from the repo's ``tiny_agent`` package, not the book's. ``tools`` is a ``LearnedTools`` registry (the book's ``NativeTools``) with the repo's ``add``, ``subtract`` and ``multiply`` functions; the description strings are kept as in the book, but a learned registry sends each function's schema (name, docstring, typed parameters), not the description. ``react`` is ``NativeReAct(max_steps=10)``. ``agent.run(query)`` loops: each reply with a tool call is executed and its result goes back as a ``tool`` message; the first reply without a tool call is the answer (``is_done``), so there is no ``final_answer`` tool. ``show_steps`` prints each step of the run from ``agent.trajectory``: the tool call, or the answer.

**What to expect.** (4.6 + 6.685) x 4 - 3.14 = 42.0, from three tool steps (add, multiply, subtract) and one answer step; the floats may carry rounding noise and the model phrases the answer freely. ``llm`` is ``qwen3.5-4b`` via the gateway, without thinking. Compare ``react``, the text-based version of the same task, and ``tiny_agent/components/planning/native_react.py``.
"""

from pathlib import Path

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.functions.add import add
from tiny_agent.components.tools.functions.multiply import multiply
from tiny_agent.components.tools.functions.subtract import subtract
from tiny_agent.components.tools.learned import LearnedTools

# Qwen 3.5 4B, served by the Docker stack (llama.cpp behind the gateway), not Ollama.
# The key is read at runtime from the git-ignored Docker secret and never printed.
api_key = (Path(__file__).parents[1] / "chat-app" / "secrets" / "llama_api_key").read_text().strip()
llm = LLM(model="qwen3.5-4b", base_url="http://localhost:3200/v1", api_key=api_key)

# The book's multi-step task: 4.6 + 6.685 = 11.285, x 4 = 45.14, - 3.14 = 42.0
query = "What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?"


def show_steps(agent):
    """Print each step of the agent's first run: the tool call it made, or its answer."""
    for index, step in enumerate(agent.trajectory.runs[0]["steps"]):
        print(f"-- Step {index+1} ---")
        if step.action:
            print(f"Tool: {step.action}")
        else:
            print(f"Answer: {step.answer}")


# Register tools (LearnedTools is the book's NativeTools: schemas in the request, structured calls back)
tools = LearnedTools()
tools.add_tool("add", add, "add(a: str, b: str)")
tools.add_tool("subtract", subtract, "subtract(a: str, b: str)")
tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

# Memory
memory = Memory()

# ReAct
react = NativeReAct(max_steps=10)

# Create agent
agent = TinyAgent(llm=llm, tools=tools, memory=memory, planner=react)


if __name__ == "__main__":
    # Multi-step task with reasoning (needs the Docker stack with qwen3.5-4b loaded)
    print(agent.run(query))
    print()
    show_steps(agent)
