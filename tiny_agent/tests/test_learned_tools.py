"""Tests for the two ways to tell a model about its tools: prompting (``PromptingTools``) and learned (``LearnedTools``)."""

from typing import Any

import pytest

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.components.tools.multiply import multiply
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.components.tools.tools import BaseTools

DESCRIPTION = "Multiplies two numbers: multiply(a: str, b: str)"
CALL = {"id": "call_1", "type": "function", "function": {"name": "multiply", "arguments": '{"a": "6", "b": "7"}'}}


class ScriptedLLM(LLM):
    """Replies with the given responses in order and records every call."""

    def __init__(self, *replies: Response):
        super().__init__(model="scripted")
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        self.calls.append((list(messages), tools))
        return self.replies.pop(0)


def make_native(**kwargs) -> LearnedTools:
    tools = LearnedTools(**kwargs)
    tools.add_tool("multiply", multiply, DESCRIPTION)
    return tools


def test_both_variants_are_tools_registries():
    assert issubclass(PromptingTools, BaseTools) and issubclass(LearnedTools, BaseTools)


# ---- how the model is told: prompt versus schemas ---------------------------------------


def test_prompting_variant_writes_the_tools_into_the_prompt_and_sends_no_schemas():
    tools = PromptingTools()
    tools.add_tool("multiply", multiply, DESCRIPTION)
    assert "`multiply`" in tools.prompt
    assert tools.schemas is None


def test_learned_variant_has_no_tool_text_in_the_prompt():
    assert make_native().prompt == ""


def test_learned_variant_sends_the_definitions_as_openai_function_schemas():
    assert make_native().schemas == [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]


def test_a_parameter_with_a_default_is_not_required():
    tools = LearnedTools()
    tools.add_tool("greet", lambda name, greeting="Hi": f"{greeting} {name}", "greets")
    parameters = tools.schemas[0]["function"]["parameters"]
    assert set(parameters["properties"]) == {"name", "greeting"}
    assert parameters["required"] == ["name"]


def test_schemas_of_an_empty_registry_is_an_empty_list():
    assert LearnedTools().schemas == []


# ---- reading the call ----------------------------------------------------------------------


def test_parse_turns_the_structured_call_into_tool_and_kwargs():
    parsed = make_native().parse(Response(content=None, tool_call=CALL, metadata={"model": "m"}))
    assert parsed.tool_call == {"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}
    assert parsed.metadata == {"model": "m"}


def test_parse_accepts_arguments_that_are_already_an_object():
    call = {"function": {"name": "multiply", "arguments": {"a": "2", "b": "3"}}}
    assert make_native().parse(Response(tool_call=call)).tool_call == {
        "tool": "multiply",
        "kwargs": {"a": "2", "b": "3"},
    }


def test_parse_accepts_a_call_without_arguments():
    call = {"function": {"name": "ping"}}
    assert make_native().parse(Response(tool_call=call)).tool_call == {"tool": "ping", "kwargs": {}}


def test_parse_leaves_a_plain_answer_alone():
    response = Response(content="It is 42.")
    assert make_native().parse(response) is response


def test_parse_ignores_json_written_in_the_text():
    # The prompting variant reads JSON out of the text; the learned one only trusts the structured call
    response = Response(content='{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}')
    assert make_native().parse(response).tool_call is None


def test_is_done_only_without_a_tool_call():
    tools = make_native()
    assert tools.is_done(Response(content="42"))
    assert not tools.is_done(tools.parse(Response(content=None, tool_call=CALL)))


def test_observation_is_a_user_message():
    assert make_native().observation(42.0) == ("user", "OBSERVATION: 42.0")


# ---- wired into TinyAgent ------------------------------------------------------------------


def test_agent_sends_the_schemas_with_every_call_and_no_tool_text_in_memory():
    tools, memory = make_native(), Memory()
    llm = ScriptedLLM(Response(content="Hello!"))
    TinyAgent(llm, memory, tools).run("Hi")
    messages, sent_tools = llm.calls[0]
    assert sent_tools == tools.schemas
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant.\n\n"}


def test_prompting_agent_sends_no_schemas():
    tools = PromptingTools()
    tools.add_tool("multiply", multiply, DESCRIPTION)
    llm = ScriptedLLM(Response(content="Hello!"))
    TinyAgent(llm, Memory(), tools).run("Hi")
    assert llm.calls[0][1] is None


def test_a_learned_tool_call_runs_the_tool_and_returns_the_observation():
    memory = Memory()
    agent = TinyAgent(ScriptedLLM(Response(content=None, tool_call=CALL)), memory, make_native())
    assert agent.run("What is 6 times 7?") == "OBSERVATION: 42.0"
    assert memory.get_messages()[-1] == {"role": "user", "content": "OBSERVATION: 42.0"}


def test_a_learned_tool_call_is_recorded_in_the_trajectory():
    agent = TinyAgent(ScriptedLLM(Response(content=None, tool_call=CALL)), Memory(), make_native())
    agent.run("What is 6 times 7?")
    (step,) = agent.trajectory.runs[0]["steps"]
    assert step.action == {"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}
    assert step.observation == "OBSERVATION: 42.0"


def test_a_plain_reply_is_the_answer_with_learned_tools():
    agent = TinyAgent(ScriptedLLM(Response(content="Hello!")), Memory(), make_native())
    assert agent.run("Hi") == "Hello!"


def test_unknown_learned_tool_is_reported_not_raised():
    call = {"function": {"name": "divide", "arguments": "{}"}}
    agent = TinyAgent(ScriptedLLM(Response(content=None, tool_call=call)), Memory(), make_native())
    assert agent.run("x") == "OBSERVATION: Tool 'divide' not found."


def test_learned_tool_needing_approval_is_denied_unless_the_user_says_yes(monkeypatch):
    tools = make_native(requires_approval=["multiply"])
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    denied = TinyAgent(ScriptedLLM(Response(content=None, tool_call=CALL)), Memory(), tools).run("x")
    assert denied == "OBSERVATION: Tool 'multiply' was denied by the user."
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert TinyAgent(ScriptedLLM(Response(content=None, tool_call=CALL)), Memory(), tools).run("x") == (
        "OBSERVATION: 42.0"
    )


def test_both_variants_reach_the_same_observation():
    prompted = PromptingTools()
    prompted.add_tool("multiply", multiply, DESCRIPTION)
    text_call = '{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}'
    a = TinyAgent(ScriptedLLM(Response(content=text_call)), Memory(), prompted).run("x")
    b = TinyAgent(ScriptedLLM(Response(content=None, tool_call=CALL)), Memory(), make_native()).run("x")
    assert a == b == "OBSERVATION: 42.0"


def test_malformed_arguments_raise_a_json_error():
    call = {"function": {"name": "multiply", "arguments": "{not json"}}
    with pytest.raises(ValueError):
        make_native().parse(Response(tool_call=call))
