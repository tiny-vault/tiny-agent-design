"""Tests for the tools: the registry and the example tools, prompting (JSON in text) and learned (structured call)
tool use, and how both are wired into TinyAgent.
"""

import pytest

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.tools.default import default_native_tools, default_tools
from tiny_agent.components.tools.functions.divide import divide
from tiny_agent.components.tools.functions.multiply import multiply
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.components.tools.tools import BaseTools
from tiny_agent.tests.stubs import ScriptedLLM

# ====================================================================================================
# test_tools
# Tool registry wiring, parsing, approval and dispatch.
#
# The model requests a named action; Python executes the registered callable.
# Scripted replies isolate that boundary so a multiplication result can be
# traced from text JSON through the registry to memory and trajectory.
#
# What this file demonstrates:
# Tests for the tool registry, the ``multiply`` tool and the tools wired into ``TinyAgent``.
#
# Expected observations and limits:
# Assertions cover numeric conversion, registration, parsing, denial and tool
# selection flags. The observation is not a final model answer; another run
# reads it. These checks do not sandbox arbitrary tools or prove robust parsing
# of every possible model response.
# ====================================================================================================

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


def test_parse_then_execute_runs_the_tool_without_an_agent():
    # A fake reply, parsed into a tool call and executed by hand: the two steps the agent does for you
    parsed = make_tools().parse(Response(content='{"tool": "multiply", "kwargs": {"a": "3.1", "b": "6.5"} }'))
    assert parsed.tool_call == {"tool": "multiply", "kwargs": {"a": "3.1", "b": "6.5"}}
    assert make_tools().execute(parsed) == pytest.approx(20.15)


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
    sent = llm.calls[1]
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


# ====================================================================================================
# test_learned_tools
# Compare prompt tool definitions with native tool schemas.
#
# Tool choice and tool execution are separate operations. A scripted model
# provides either JSON text or a structured call, so both registries can be
# compared on the same multiplication without stochastic model behavior.
#
# What this file demonstrates:
# Tests for the two ways to tell a model about its tools: prompting (``PromptingTools``) and learned (``LearnedTools``).
#
# Expected observations and limits:
# ``CALL`` is the server-shaped request and ``DESCRIPTION`` names the tool.
# Assertions check schemas, parsed arguments, approval, observations and agent
# trajectory. Both variants returning 42 proves matching dispatch for these
# inputs, not that one prompting or training approach is superior.
# ====================================================================================================

DESCRIPTION = "Multiplies two numbers: multiply(a: str, b: str)"
NATIVE_CALL = {"id": "call_1", "type": "function", "function": {"name": "multiply", "arguments": '{"a": "6", "b": "7"}'}}


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
                    "properties": {
                        "a": {"type": "number", "description": 'The first factor, a number such as 6 (or numeric text such as "6").'},
                        "b": {"type": "number", "description": 'The second factor, a number such as 7 (or numeric text such as "7").'},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]


def test_default_native_tools_advertise_number_parameters():
    properties = default_native_tools().schemas[0]["function"]["parameters"]["properties"]
    assert {name: p["type"] for name, p in properties.items()} == {"a": "number", "b": "number"}


def test_a_parameter_type_comes_from_its_annotation():
    def tool(count: int, ratio: float, flag: bool, name: str, other, either: float | str):
        """Args:
        count: how many
        """

    tools = LearnedTools()
    tools.add_tool("tool", tool, "d")
    properties = tools.schemas[0]["function"]["parameters"]["properties"]
    assert {n: p["type"] for n, p in properties.items()} == {
        "count": "integer", "ratio": "number", "flag": "boolean", "name": "string", "other": "string", "either": "number",
    }  # fmt: skip


def test_a_parameter_description_comes_from_the_docstring_args_section():
    def tool(a: str, b: str):
        """Does a thing.

        Arguments:
            a: the first one,
                continued here.
            b: the second one.

        Returns:
            Something: not a parameter.
        """

    tools = LearnedTools()
    tools.add_tool("tool", tool, "d")
    properties = tools.schemas[0]["function"]["parameters"]["properties"]
    assert properties["a"]["description"] == "the first one,"
    assert properties["b"]["description"] == "the second one."
    assert "Something" not in str(properties)


def test_a_parameter_without_a_docstring_entry_has_no_description():
    tools = LearnedTools()
    tools.add_tool("tool", lambda x: x, "d")
    assert tools.schemas[0]["function"]["parameters"]["properties"] == {"x": {"type": "string"}}


def test_a_number_schema_still_runs_with_numbers_and_with_text():
    assert multiply(6, 7) == 42.0
    assert multiply("6", "7") == 42.0
    assert divide(42, 6) == 7.0


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
    parsed = make_native().parse(Response(content=None, tool_call=NATIVE_CALL, metadata={"model": "m"}))
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


def test_native_parse_leaves_a_plain_answer_alone():
    response = Response(content="It is 42.")
    assert make_native().parse(response) is response


def test_parse_ignores_json_written_in_the_text():
    # The prompting variant reads JSON out of the text; the learned one only trusts the structured call
    response = Response(content='{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}')
    assert make_native().parse(response).tool_call is None


def test_is_done_only_without_a_tool_call():
    tools = make_native()
    assert tools.is_done(Response(content="42"))
    assert not tools.is_done(tools.parse(Response(content=None, tool_call=NATIVE_CALL)))


def test_a_final_answer_call_is_done_and_its_kwargs_become_the_answer():
    tools = make_native()
    response = Response(content=None, tool_call={"tool": "final_answer", "kwargs": "It is 42."})
    assert tools.is_done(response)
    assert response.content == "It is 42."
    # Anything that is not text is turned into text
    response = Response(content=None, tool_call={"tool": "final_answer", "kwargs": {"answer": 42}})
    assert tools.is_done(response)
    assert response.content == "{'answer': 42}"


def test_the_agent_returns_the_final_answer_and_runs_no_tool():
    call = '{"tool": "final_answer", "kwargs": "It is 42."}'
    memory = Memory()
    agent = TinyAgent(ScriptedLLM(call), memory, make_tools())
    assert agent.run("What is 6 times 7?") == "It is 42."
    # The raw reply stays in memory, and no OBSERVATION was added
    assert memory.get_messages()[-1] == {"role": "assistant", "content": call}
    (step,) = agent.trajectory.runs[0]["steps"]
    assert (step.answer, step.action, step.observation) == ("It is 42.", None, None)


def test_observation_is_a_tool_message():
    assert make_native().observation(42.0) == ("tool", "OBSERVATION: 42.0")


# ---- wired into TinyAgent ------------------------------------------------------------------


def test_agent_sends_the_schemas_with_every_call_and_no_tool_text_in_memory():
    tools, memory = make_native(), Memory()
    llm = ScriptedLLM(Response(content="Hello!"))
    TinyAgent(llm, memory, tools).run("Hi")
    messages, sent_tools = llm.calls[0], llm.tools[0]
    assert sent_tools == tools.schemas
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant.\n\n"}


def test_prompting_agent_sends_no_schemas():
    tools = PromptingTools()
    tools.add_tool("multiply", multiply, DESCRIPTION)
    llm = ScriptedLLM(Response(content="Hello!"))
    TinyAgent(llm, Memory(), tools).run("Hi")
    assert llm.tools[0] is None


def test_a_learned_tool_call_runs_the_tool_and_returns_the_observation():
    memory = Memory()
    agent = TinyAgent(ScriptedLLM(Response(content=None, tool_call=NATIVE_CALL)), memory, make_native())
    assert agent.run("What is 6 times 7?") == "OBSERVATION: 42.0"
    # The model's own call is kept with its message, and the result follows it as a `tool` message
    assert memory.get_messages()[-2] == {"role": "assistant", "content": "", "tool_calls": [NATIVE_CALL]}
    assert memory.get_messages()[-1] == {"role": "tool", "content": "OBSERVATION: 42.0"}


def test_a_prompted_tool_call_is_not_stored_as_a_structured_call():
    memory = Memory()
    TinyAgent(ScriptedLLM(CALL), memory, make_tools()).run("What is 6 times 7?")
    assert "tool_calls" not in memory.get_messages()[-2]


def test_a_learned_tool_call_is_recorded_in_the_trajectory():
    agent = TinyAgent(ScriptedLLM(Response(content=None, tool_call=NATIVE_CALL)), Memory(), make_native())
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
    denied = TinyAgent(ScriptedLLM(Response(content=None, tool_call=NATIVE_CALL)), Memory(), tools).run("x")
    assert denied == "OBSERVATION: Tool 'multiply' was denied by the user."
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert TinyAgent(ScriptedLLM(Response(content=None, tool_call=NATIVE_CALL)), Memory(), tools).run("x") == (
        "OBSERVATION: 42.0"
    )


def test_both_variants_reach_the_same_observation():
    prompted = PromptingTools()
    prompted.add_tool("multiply", multiply, DESCRIPTION)
    text_call = '{"tool": "multiply", "kwargs": {"a": "6", "b": "7"}}'
    a = TinyAgent(ScriptedLLM(Response(content=text_call)), Memory(), prompted).run("x")
    b = TinyAgent(ScriptedLLM(Response(content=None, tool_call=NATIVE_CALL)), Memory(), make_native()).run("x")
    assert a == b == "OBSERVATION: 42.0"


def test_malformed_arguments_raise_a_json_error():
    call = {"function": {"name": "multiply", "arguments": "{not json"}}
    with pytest.raises(ValueError):
        make_native().parse(Response(tool_call=call))
