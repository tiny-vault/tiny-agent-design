"""ReAct (Reason and Act): the prompt and parser that make an agent think, act, observe and repeat.

A single model call can answer or request one tool, but a task that needs several
tools needs a cycle. ReAct (Yao et al., 2022) is that cycle in the prompt: each turn
the model writes one ``THOUGHT`` (its reasoning) and one ``ACTION`` (a tool call as
JSON), the harness runs the action and replies with an ``OBSERVATION``, and the model
goes again until it calls the ``final_answer`` tool.

``ReAct.prompt`` teaches that format and goes into the system message before the tool
prompt. ``ReAct.parse`` splits a reply into its two sections: the ``ACTION`` text
becomes ``content`` (what ``PromptingTools.parse`` then reads the JSON from) and the
``THOUGHT`` becomes ``reasoning`` (what the trajectory records as the step's thought). ``reasoning``
is also where ``LLM.generate`` puts the server's own thinking (``think=True``): when that is present the
two are joined, native thinking first, so neither erases the other.
Reading stops at an ``OBSERVATION:`` marker, so an observation the model invents is
never used. A reply with no ``ACTION:`` is left as it is, so a plain answer is still
an answer. ``max_steps`` is how many turns ``TinyAgent.run`` may take.

The parser is a text heuristic: it does not check that the action is valid JSON, and
it only fits prompted tools (``PromptingTools``), because ``LearnedTools`` reads a
structured call, not text. Compare ``tools.py`` for ``final_answer`` and ``is_done``.
"""

import re

from tiny_agent.components.client.response import Response


class ReAct:
    """ReAct module: the prompt that asks for THOUGHT and ACTION, and the parser that reads them."""

    def __init__(self, max_steps: int = 10):
        """Initialize the ReAct module.

        Args:
            max_steps: The most turns the agent may take on one task before it gives up.
        """
        self.max_steps = max_steps

    @property
    def prompt(self) -> str:
        """The system-prompt section that teaches the THOUGHT / ACTION format and the ``final_answer`` call."""
        return """
# ReAct (Reason and Act)

You are a ReAct agent that performs exactly ONE step per turn.

## ReAct Format

You use the following format for each step:

THOUGHT: [Your reasoning about what to do next]
ACTION:
{
    "tool": "a_tool_name",
    "kwargs": {"param": "value"}
}

An observation will be provided after each action. You do not generate the observation yourself.

## ReAct Completion

To provide the final answer to the task, use an action blob with "tool": "final_answer" tool.
It is the only way to complete the task, else you will be stuck on a loop.
So your final output should look like this:

ACTION:
{
    "tool": "final_answer",
    "kwargs": "insert your final answer here"
}

The "kwargs" of `final_answer` is the answer written as a plain string, never an object such as {"answer": "..."}.
Use the `final_answer` tool when you are completely done with all subtasks and have the final answer ready.
You can also use `final_answer` to directly reply to a user's question without using any other tools.

"""

    def parse(self, response: Response) -> Response:
        """Parse a ReAct formatted reply into THOUGHT and ACTION.

        Args:
            response: The model's reply. It is changed in place: ``content`` becomes the ACTION
                text and ``reasoning`` the THOUGHT text.

        Returns:
            The same response. When the reply has no ``ACTION:`` it is returned unchanged.
        """
        text = response.content or ""

        # The patterns for each section; both stop at OBSERVATION: to ignore a hallucinated observation
        patterns = {
            "THOUGHT": r"THOUGHT:\s*(.+?)(?=ACTION:|OBSERVATION:|$)",
            "ACTION": r"ACTION:\s*(.+?)(?=THOUGHT:|OBSERVATION:|$)",
        }

        # Extract each section using regex
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            result[key] = match.group(1).strip() if match else ""

        # No action: a plain reply, keep it as the answer instead of replacing it with ""
        if not result["ACTION"]:
            return response

        # Update the response and keep only the action as its content
        response.content = result["ACTION"]
        # reasoning may already hold the model's native thinking (think=True): keep it and add the THOUGHT
        response.reasoning = "\n\n".join(p for p in (response.reasoning, result["THOUGHT"]) if p) or None
        return response
