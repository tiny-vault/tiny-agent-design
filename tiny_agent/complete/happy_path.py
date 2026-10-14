"""The complete happy path of TinyAgent on the real Docker model: one tool by prompting, then one by learned calling.

What is it? A model can be told about its tools in two ways, and this run uses both, one after the other, on the
model server of ``node execute.mjs up llm-client`` (``TINYAGENT_LLM_URL``, default ``http://127.0.0.1:3200/v1``;
load a chat model with ``node execute.mjs llm-client load qwen3.5-4b``). Nothing is faked.

1. **Prompting:** ``PromptingTools`` writes the tool definitions into the system prompt and the model answers with
   JSON text (``{"tool": "multiply", "kwargs": {...}}``). The task is ``6 times 7``; the observation is ``42.0``.
2. **Learned:** ``LearnedTools`` leaves the prompt empty and sends the definitions in the request's ``tools`` field;
   the model answers with a structured tool call. The task takes the first stage's result and divides it by 6, so the
   second stage depends on the first; the observation is ``7.0``.

Each stage is its own agent with its own ``Memory``, and each ends with a second ``run`` so the model reads the
observation and states the answer (the agent has no loop yet, so a tool call and its answer are two runs).

Chat replies vary from run to run, so the checks look for the facts that must be there (the tool called, the number)
and never for exact wording. A failed check raises ``AssertionError``.

Run it with ``python -m tiny_agent.complete.happy_path``.
"""

import sys
from typing import Any

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client import check
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.tools.functions.divide import divide
from tiny_agent.components.tools.functions.multiply import multiply
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.config.settings import Settings

ANSWER_PROMPT = "In one short sentence, what was the result?"


class StackNotRunning(RuntimeError):
    """No chat model answers on the model server; the message says which command starts it."""


def real_llm() -> LLM:
    """A client for the chat model that is loaded now (``TINYAGENT_MODEL`` if set).

    Raises:
        StackNotRunning: The server is down or has no model loaded.
    """
    settings = Settings.from_env()
    loaded = check.loaded_models(settings.llm_url)
    if not loaded:
        raise StackNotRunning(
            f"no chat model loaded at {settings.llm_url}; run: node execute.mjs up llm-client, "
            "then node execute.mjs llm-client load qwen3.5-4b"
        )
    api_key = settings.resolve_api_key()
    key: dict[str, Any] = {"api_key": api_key} if api_key else {}
    return LLM(model=settings.model or loaded[0], base_url=settings.llm_url, **key)


def require(condition: object, message: str) -> None:
    """Fail with ``message`` unless ``condition`` holds."""
    if not condition:
        raise AssertionError(message)


def run_stage(agent: TinyAgent, task: str, tool: str, observation: str) -> dict[str, Any]:
    """Run one tool round trip on ``agent`` and check every part of it.

    Run 1 is the task: the model asks for ``tool``, the harness runs it and returns the observation. Run 2 is a
    follow-up: the observation is in memory, so the model can state the result.

    Returns:
        What happened: ``prompt`` (the first message), ``action``, ``observation`` and ``answer``.
    """
    first_message = agent.memory.get_messages()[0]
    result = agent.run(task)
    require(result == observation, f"running {tool} should return {observation!r}, got {result!r}")

    (step,) = agent.trajectory.runs[0]["steps"]
    require(step.action and step.action["tool"] == tool, f"the step should record a call to {tool}: {step.action}")
    require(step.observation == observation and step.answer is None, "the step holds the observation, not an answer")
    # Learned tools answer the recorded tool call with a `tool` message; prompted ones with a `user` message
    role = "tool" if isinstance(agent.tools, LearnedTools) else "user"
    require(agent.memory.get_messages()[-1] == {"role": role, "content": observation}, "the observation is in memory")

    answer = agent.run(ANSWER_PROMPT)
    number = observation.removeprefix("OBSERVATION: ").removesuffix(".0")
    require(number in answer, f"the model should read the observation and state {number}: {answer!r}")
    return {"prompt": first_message["content"], "action": step.action, "observation": observation, "answer": answer}


def prompting_stage(llm: LLM) -> dict[str, Any]:
    """Stage 1: ``multiply`` told to the model in the system prompt, called with JSON text."""
    tools = PromptingTools()
    tools.add_tool(
        "multiply",
        multiply,
        "Multiplies two numbers and returns the product as a decimal number. Use it for every multiplication "
        "instead of working it out yourself. Both arguments are numbers written as text, such as \"6\"; text that "
        "is not a number is an error: multiply(a: str, b: str). Example: multiply(\"6\", \"7\") returns 42.0.",
    )
    require(tools.schemas is None, "prompting tools send no `tools` field in the request")
    agent = TinyAgent(llm, Memory(), tools)
    require("multiply" in agent.memory.get_messages()[0]["content"], "the tool is described in the system prompt")
    return run_stage(agent, "Use the multiply tool with a=\"6\" and b=\"7\".", "multiply", "OBSERVATION: 42.0")


def learned_stage(llm: LLM, previous: str) -> dict[str, Any]:
    """Stage 2: ``divide`` sent in the request's ``tools`` field, called with a structured tool call.

    Args:
        llm: The model.
        previous: The observation of stage 1; its number is what gets divided.
    """
    tools = LearnedTools()
    tools.add_tool(
        "divide",
        divide,
        "Divides a by b and returns the quotient as a decimal number. Use it for every division instead of "
        "working it out yourself. a is the dividend and b the divisor; b must not be zero, and text that is "
        "not a number is an error. Example: a=42, b=6 returns 7.0.",
    )
    (schema,) = tools.schemas
    require(schema["function"]["name"] == "divide", "the definition travels as a schema")
    require(schema["function"]["parameters"]["required"] == ["a", "b"], "both parameters are required")
    agent = TinyAgent(llm, Memory(), tools)
    require("divide" not in agent.memory.get_messages()[0]["content"], "the system prompt does not mention the tool")
    number = previous.removeprefix("OBSERVATION: ")
    return run_stage(agent, f"Use the divide tool with a=\"{number}\" and b=\"6\".", "divide", "OBSERVATION: 7.0")


def run_happy_path(llm: LLM | None = None) -> dict[str, dict[str, Any]]:
    """Run the prompting stage, then the learned stage that builds on its result.

    Args:
        llm: The model. Defaults to :func:`real_llm`.

    Returns:
        ``{"prompting": ..., "learned": ...}``, what each stage found.

    Raises:
        StackNotRunning: There is no model to talk to.
        AssertionError: A check failed.
    """
    llm = llm or real_llm()
    prompting = prompting_stage(llm)
    return {"prompting": prompting, "learned": learned_stage(llm, prompting["observation"])}


def main() -> int:
    """Run the happy path, print what each stage did, and return the exit code."""
    try:
        stages = run_happy_path()
    except StackNotRunning as error:
        print(f"cannot run: {error}")
        return 2
    for name, facts in stages.items():
        print(f"{name} tools")
        print(f"  action:      {facts['action']}")
        print(f"  observation: {facts['observation']}")
        print(f"  answer:      {facts['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
