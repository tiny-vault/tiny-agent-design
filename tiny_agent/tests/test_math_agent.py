"""Tests for the math agent: gemma4-e4b with thinking, three native arithmetic tools, a NativeReAct loop."""

from tiny_agent.agents.math_agent import MODEL, create_math_agent
from tiny_agent.components.client.response import Response
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.tests.stubs import ScriptedLLM

ADD = {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 4.6, "b": 6.685}'}}


def test_default_llm_is_gemma_with_native_thinking():
    agent = create_math_agent()
    assert (agent.llm.model, agent.llm.think, MODEL) == ("gemma4-e4b", True, "gemma4-e4b")


def test_math_agent_has_three_native_tools_and_native_react():
    agent = create_math_agent(ScriptedLLM())
    assert isinstance(agent.tools, LearnedTools) and isinstance(agent.planner, NativeReAct)
    assert [schema["function"]["name"] for schema in agent.tools.schemas] == ["add", "subtract", "multiply"]
    # NativeReAct and learned tools add no text: the system prompt is only the greeting
    assert agent.memory.get_messages() == [{"role": "system", "content": "You are a helpful assistant.\n\n"}]


def test_each_call_builds_a_new_agent_with_empty_memory():
    first, second = create_math_agent(ScriptedLLM()), create_math_agent(ScriptedLLM())
    assert first is not second and first.memory is not second.memory


def test_math_agent_calls_a_tool_then_answers():
    llm = ScriptedLLM(
        Response(content=None, reasoning="Add first.", tool_call=ADD),
        Response(content="The sum is 11.285.", reasoning="Done."),
    )
    agent = create_math_agent(llm)
    assert agent.run("What is 4.6 + 6.685?") == "The sum is 11.285."
    first, second = agent.trajectory.runs[0]["steps"]
    assert (first.thought, first.action) == ("Add first.", {"tool": "add", "kwargs": {"a": 4.6, "b": 6.685}})
    assert first.observation is not None and first.observation.startswith("OBSERVATION: 11.28")
    assert (second.thought, second.answer) == ("Done.", "The sum is 11.285.")
