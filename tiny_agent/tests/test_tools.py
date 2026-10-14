"""Tests for the tool registry, the ``multiply`` tool and the tools wired into ``TinyAgent``."""

from typing import Any

import pytest

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.tools.default import default_tools
from tiny_agent.components.tools.multiply import multiply
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.components.tools.tools import BaseTools


def test_multiply_takes_text_and_returns_a_float():
    assert multiply("3", "4.5") == 13.5


def test_add_tool_registers_function_and_description():
    tools = PromptingTools()
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)",
    )
    entry = tools.registry["multiply"]
    assert entry["function"]("2", "5") == 10.0
    assert entry["description"].startswith("Multiplies")
    assert tools.schemas is None


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("2", "3", 6.0), ("-2", "3", -6.0), ("0", "99", 0.0), ("1e3", "2", 2000.0), (" 4 ", "0.5", 2.0)],
)
def test_multiply_cases(a, b, expected):
    result = multiply(a, b)
    assert isinstance(result, float)
    assert result == expected


@pytest.mark.parametrize(("a", "b"), [("abc", "2"), ("2", ""), ("", "")])
def test_multiply_rejects_text_that_is_not_a_number(a, b):
    with pytest.raises(ValueError):
        multiply(a, b)


def test_new_registry_is_empty():
    assert PromptingTools().registry == {}


def test_description_defaults_to_empty_string():
    tools = PromptingTools()
    tools.add_tool("multiply", multiply)
    assert tools.registry["multiply"]["description"] == ""


def test_add_tool_keeps_several_tools_apart():
    tools = PromptingTools()
    tools.add_tool("multiply", multiply, "multiplies")
    tools.add_tool("negate", lambda a: -float(a), "negates")
    assert set(tools.registry) == {"multiply", "negate"}
    assert tools.registry["negate"]["function"]("3") == -3.0


def test_adding_a_name_twice_replaces_the_first_tool():
    tools = PromptingTools()
    tools.add_tool("op", multiply, "old")
    tools.add_tool("op", lambda a, b: "new", "new")
    assert tools.registry["op"]["description"] == "new"
    assert tools.registry["op"]["function"]("1", "2") == "new"


def test_requires_approval_is_stored():
    assert PromptingTools(requires_approval=["multiply"]).requires_approval == ["multiply"]


# ---- the tools wired into TinyAgent ----------------------------------------------------


class ScriptedLLM(LLM):
    """Replies with the given texts in order and records the messages of every call."""

    def __init__(self, *replies: str):
        super().__init__(model="scripted")
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        self.calls.append((list(messages), tools))
        return Response(content=self.replies.pop(0))


def make_tools(**kwargs) -> PromptingTools:
    tools = PromptingTools(**kwargs)
    tools.add_tool("multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)")
    return tools


CALL = '{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}'


def test_prompt_lists_the_tools_and_the_json_format():
    prompt = make_tools().prompt
    assert "`multiply`: Multiplies two numbers" in prompt
    assert '{"tool": "name", "kwargs"' in prompt


def test_agent_puts_the_tool_prompt_in_memory_as_a_system_message():
    tools, memory = make_tools(), Memory()
    agent = TinyAgent(ScriptedLLM(), memory, tools)
    assert agent.tools is tools
    assert memory.get_messages()[0]["role"] == "system"
    assert tools.prompt in memory.get_messages()[0]["content"]


def test_agent_without_tools_adds_no_system_message():
    memory = Memory()
    TinyAgent(ScriptedLLM(), memory)
    assert memory.get_messages() == []


def test_parse_extracts_a_tool_call_from_text_around_the_json():
    parsed = PromptingTools().parse(Response(content=f"Let me compute. {CALL} done"))
    assert parsed.tool_call == {"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}


def test_parse_leaves_a_plain_answer_alone():
    response = Response(content="It is 42.")
    assert PromptingTools().parse(response) is response
    assert PromptingTools().parse(Response(content=None)).tool_call is None


def test_a_tool_call_runs_the_tool_and_returns_the_observation():
    memory = Memory()
    agent = TinyAgent(ScriptedLLM(CALL), memory, make_tools())
    assert agent.run("What is 6 times 7?") == "OBSERVATION: 42.0"
    assert memory.get_messages()[-2:] == [
        {"role": "assistant", "content": CALL},
        {"role": "user", "content": "OBSERVATION: 42.0"},
    ]


def test_a_tool_call_is_recorded_in_the_trajectory():
    agent = TinyAgent(ScriptedLLM(CALL), Memory(), make_tools())
    agent.run("What is 6 times 7?")
    (step,) = agent.trajectory.runs[0]["steps"]
    assert step.action == {"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}
    assert step.observation == "OBSERVATION: 42.0"
    assert step.answer is None


def test_a_plain_reply_is_the_answer_even_with_tools():
    agent = TinyAgent(ScriptedLLM("Hello!"), Memory(), make_tools())
    assert agent.run("Hi") == "Hello!"
    (step,) = agent.trajectory.runs[0]["steps"]
    assert (step.answer, step.action, step.observation) == ("Hello!", None, None)


def test_the_observation_reaches_the_model_on_the_next_run():
    llm = ScriptedLLM(CALL, "The answer is 42.")
    agent = TinyAgent(llm, Memory(), make_tools())
    agent.run("What is 6 times 7?")
    assert agent.run("So?") == "The answer is 42."
    sent = llm.calls[1][0]
    assert {"role": "user", "content": "OBSERVATION: 42.0"} in sent


def test_unknown_tool_is_reported_not_raised():
    call = '{"tool": "divide", "kwargs": {}}'
    assert TinyAgent(ScriptedLLM(call), Memory(), make_tools()).run("x") == "OBSERVATION: Tool 'divide' not found."


def test_tool_needing_approval_is_denied_unless_the_user_says_yes(monkeypatch):
    tools = make_tools(requires_approval=["multiply"])
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert TinyAgent(ScriptedLLM(CALL), Memory(), tools).run("x") == (
        "OBSERVATION: Tool 'multiply' was denied by the user."
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert TinyAgent(ScriptedLLM(CALL), Memory(), tools).run("x") == "OBSERVATION: 42.0"


def test_default_tools_registers_multiply():
    tools = default_tools()
    assert set(tools.registry) == {"multiply"}
    assert tools.registry["multiply"]["function"] is multiply
    assert tools.registry["multiply"]["description"].startswith("Multiplies")


def test_default_tools_passes_requires_approval_and_does_not_share_it():
    assert default_tools(["multiply"]).requires_approval == ["multiply"]
    assert default_tools().requires_approval == []
    assert default_tools().requires_approval is not default_tools().requires_approval


def test_tools_inherit_the_registry_from_base_tools():
    assert issubclass(PromptingTools, BaseTools)
    with pytest.raises(TypeError):
        BaseTools()  # pyright: ignore[reportAbstractUsage]


def test_a_registry_does_not_share_requires_approval_with_others():
    a, b = PromptingTools(), PromptingTools()
    a.requires_approval.append("multiply")
    assert b.requires_approval == []


def test_base_memory_cannot_be_created_and_every_memory_inherits_it():
    from tiny_agent.components.memory.memory import BaseMemory, Memory
    from tiny_agent.components.memory.trimming import TrimmingMemory

    with pytest.raises(TypeError):
        BaseMemory()  # pyright: ignore[reportAbstractUsage]
    assert issubclass(Memory, BaseMemory) and issubclass(TrimmingMemory, BaseMemory)


# ---- binding tools to the agent with flags ----------------------------------------------


def test_no_flag_binds_no_tools():
    from tiny_agent.components.tools.default import tools_from_settings
    from tiny_agent.config.settings import Settings

    assert tools_from_settings(Settings()) is None


def test_the_tools_flag_binds_prompting_tools_and_native_wins():
    from tiny_agent.components.tools.default import tools_from_settings
    from tiny_agent.components.tools.learned import LearnedTools
    from tiny_agent.config.settings import Settings

    prompting = tools_from_settings(Settings(tools=True))
    assert type(prompting) is PromptingTools and "multiply" in prompting.registry
    native = tools_from_settings(Settings(tools=True, native_tools=True))
    assert type(native) is LearnedTools and "multiply" in native.registry


def test_the_demo_binds_tools_when_the_flag_is_on(demo_server, monkeypatch):
    from tiny_agent import main as demo

    monkeypatch.setenv("TINYAGENT_TOOLS", "1")
    demo.main()
    first = demo_server.requests[0]["body"]
    assert first["messages"][0]["role"] == "system" and "multiply" in first["messages"][0]["content"]
    assert "tools" not in first  # prompting tools are told in the prompt, not in the request's tools field


def test_the_demo_sends_native_tools_in_the_request(demo_server, monkeypatch):
    from tiny_agent import main as demo

    monkeypatch.setenv("TINYAGENT_NATIVE_TOOLS", "1")
    demo.main()
    first = demo_server.requests[0]["body"]
    assert first["tools"][0]["function"]["name"] == "multiply"


def test_the_demo_has_no_tools_by_default(demo_server):
    from tiny_agent import main as demo

    demo.main()
    first = demo_server.requests[0]["body"]
    assert first["messages"] == [{"role": "user", "content": demo.GREETING}] and "tools" not in first
