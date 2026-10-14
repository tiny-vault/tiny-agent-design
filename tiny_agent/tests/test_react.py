"""Tests for ReAct: the prompt, the THOUGHT / ACTION parser and the agent loop it drives."""

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning import ReAct
from tiny_agent.components.tools.functions.add import add
from tiny_agent.components.tools.functions.divide import divide
from tiny_agent.components.tools.functions.multiply import multiply
from tiny_agent.components.tools.functions.subtract import subtract
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.tests.stubs import ScriptedLLM

CALL = 'THOUGHT: I need the product.\nACTION:\n{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}'
FINISH = 'THOUGHT: I have it.\nACTION:\n{"tool": "final_answer", "kwargs": "The answer is 42."}'


def make_tools() -> PromptingTools:
    tools = PromptingTools()
    tools.add_tool("multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)")
    return tools


def test_parse_splits_thought_from_action():
    parsed = ReAct().parse(Response(content=CALL))
    assert parsed.reasoning == "I need the product."
    assert parsed.content == '{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}'


def test_parse_ignores_an_invented_observation():
    parsed = ReAct().parse(Response(content=CALL + "\nOBSERVATION: 41"))
    assert (parsed.content or "").endswith("}") and "41" not in (parsed.content or "")


def test_parse_keeps_a_reply_without_an_action():
    parsed = ReAct().parse(Response(content="Hello!"))
    assert (parsed.content, parsed.reasoning) == ("Hello!", None)


def test_prompt_example_is_valid_json_and_max_steps_defaults_to_ten():
    assert ",\n}" not in ReAct().prompt
    assert ReAct().max_steps == 10


def test_prompt_asks_for_the_final_answer_as_a_plain_string():
    assert "plain string, never an object" in ReAct().prompt


def test_planner_prompt_comes_before_the_tool_prompt():
    memory, tools, planner = Memory(), make_tools(), ReAct()
    TinyAgent(ScriptedLLM(), memory, tools, planner)
    system = memory.get_messages()[0]["content"]
    assert system.index(planner.prompt) < system.index(tools.prompt)


def test_the_loop_runs_the_tool_then_returns_the_final_answer():
    llm, memory = ScriptedLLM(CALL, FINISH), Memory()
    agent = TinyAgent(llm, memory, make_tools(), ReAct())
    assert agent.run("What is 6 times 7?") == "The answer is 42."
    # The second call saw the first reply and its observation
    assert {"role": "user", "content": "OBSERVATION: 42.0"} in llm.calls[1]
    first, second = agent.trajectory.runs[0]["steps"]
    assert (first.thought, first.observation) == ("I need the product.", "OBSERVATION: 42.0")
    assert (second.answer, second.action) == ("The answer is 42.", None)


def test_a_plain_reply_ends_the_loop():
    assert TinyAgent(ScriptedLLM("Hello!"), Memory(), make_tools(), ReAct()).run("Hi") == "Hello!"


def test_the_loop_stops_at_max_steps():
    llm = ScriptedLLM(CALL, CALL, CALL)
    agent = TinyAgent(llm, Memory(), make_tools(), ReAct(max_steps=2))
    assert agent.run("x") == "Max steps reached without completion."
    assert len(llm.calls) == 2


def test_without_a_planner_run_is_still_one_step():
    llm = ScriptedLLM(CALL, FINISH)
    assert TinyAgent(llm, Memory(), make_tools()).run("x").startswith("OBSERVATION")
    assert len(llm.calls) == 1


def test_a_multi_step_task_uses_three_tools_in_turn():
    tools = PromptingTools()
    tools.add_tool("add", add, "add(a: str, b: str)")
    tools.add_tool("subtract", subtract, "subtract(a: str, b: str)")
    tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

    def act(thought: str, tool: str, a: str, b: str) -> str:
        return f'THOUGHT: {thought}\nACTION:\n{{"tool": "{tool}", "kwargs": {{"a": "{a}", "b": "{b}"}}}}'

    llm = ScriptedLLM(
        act("First the sum.", "add", "4.6", "6.685"),
        act("Now times four.", "multiply", "11.285", "4"),
        act("Now subtract.", "subtract", "45.14", "3.14"),
        'THOUGHT: Done.\nACTION:\n{"tool": "final_answer", "kwargs": "42.0"}',
    )
    agent = TinyAgent(llm=llm, tools=tools, memory=Memory(), planner=ReAct(max_steps=10))
    assert agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?") == "42.0"
    steps = agent.trajectory.runs[0]["steps"]
    assert [s.action["tool"] for s in steps[:3]] == ["add", "multiply", "subtract"]
    assert steps[2].observation == "OBSERVATION: 42.0"


def test_a_failing_tool_becomes_an_observation_and_the_loop_goes_on():
    divide_by_zero = 'ACTION:\n{"tool": "divide", "kwargs": {"a": "1", "b": "0"}}'
    tools = make_tools()
    tools.add_tool("divide", divide, "divide(a: str, b: str)")
    llm = ScriptedLLM(divide_by_zero, FINISH)
    agent = TinyAgent(llm, Memory(), tools, ReAct())
    assert agent.run("x") == "The answer is 42."
    assert {"role": "user", "content": "OBSERVATION: Error: ZeroDivisionError: float division by zero"} in llm.calls[1]


def test_a_malformed_action_becomes_an_observation_and_the_loop_goes_on():
    llm = ScriptedLLM('ACTION:\n{"tool": "multiply", "kwargs": {"a": "6",}}', FINISH)
    agent = TinyAgent(llm, Memory(), make_tools(), ReAct())
    assert agent.run("x") == "The answer is 42."
    assert "could not read your ACTION" in llm.calls[1][-1]["content"]


def test_without_a_planner_a_failing_tool_still_raises():
    import pytest

    call = 'ACTION:\n{"tool": "multiply", "kwargs": {"a": "x", "b": "1"}}'
    with pytest.raises(ValueError):
        TinyAgent(ScriptedLLM(call), Memory(), make_tools()).run("x")


def test_hitting_max_steps_is_the_last_step_of_the_trajectory():
    agent = TinyAgent(ScriptedLLM(CALL, CALL), Memory(), make_tools(), ReAct(max_steps=2))
    agent.run("x")
    steps = agent.trajectory.runs[0]["steps"]
    assert len(steps) == 3
    assert steps[-1].answer == "Max steps reached without completion."


def test_a_trimming_memory_keeps_the_task_through_a_long_loop():
    from tiny_agent.components.memory.trimming import TrimmingMemory

    llm = ScriptedLLM(CALL, CALL, CALL, FINISH)
    memory = TrimmingMemory()
    TinyAgent(llm, memory, make_tools(), ReAct()).run("What is 6 times 7?")
    # By the last call the window has slid past the first replies, yet the task is still sent
    assert {"role": "user", "content": "What is 6 times 7?"} in llm.calls[-1]
    assert len(llm.calls[-1]) <= 1 + 1 + 4  # system, pinned task, last four


def test_the_next_run_pins_its_own_task():
    from tiny_agent.components.memory.trimming import TrimmingMemory

    llm = ScriptedLLM(FINISH, CALL, CALL, CALL, FINISH)
    memory = TrimmingMemory()
    agent = TinyAgent(llm, memory, make_tools(), ReAct())
    agent.run("first")
    agent.run("second")
    contents = [m["content"] for m in llm.calls[-1]]
    assert "second" in contents and "first" not in contents


def test_native_thinking_and_the_thought_are_both_kept():
    both = ReAct().parse(Response(content=CALL, reasoning="native thinking"))
    assert both.reasoning == "native thinking\n\nI need the product."
    # No THOUGHT in the reply: the native thinking stays, and nothing replaces None with ""
    assert ReAct().parse(Response(content='ACTION:\n{"tool": "x"}', reasoning="native")).reasoning == "native"
    assert ReAct().parse(Response(content='ACTION:\n{"tool": "x"}')).reasoning is None
