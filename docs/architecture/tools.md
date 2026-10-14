# Tools

A tool is a plain Python function the agent may call to affect the world. The model never runs anything itself: it only writes text. The harness reads a tool call out of that text, runs the function, and shows the result to the model as an *observation*.

```{mermaid}
flowchart LR
    L["LLM.generate"] -->|reply| P["tools.parse"]
    P -->|tool_call| X["tools.execute"]
    X -->|result| O["tools.observation"]
    O -->|"user message: OBSERVATION: ..."| M["memory"]
    P -->|no tool call| D["is_done: the reply is the answer"]
```

## The registry

`BaseTools` ({py:class}`tiny_agent.components.tools.tools.BaseTools`) holds the registered tools. `add_tool(name, func, description)` registers one, `execute` runs it, and tools named in `requires_approval` ask the user first (`Allow multiply? [y/N]`). An unknown tool, or a denied one, comes back as a message instead of an exception, so the model can read what went wrong.

The two variants below share all of that. They differ only in how the model is told about the tools and how its call is read back.

## Prompted or learned

| | `PromptingTools` | `LearnedTools` |
|---|---|---|
| For a model that | is told about its tools in the prompt | learned tool calling in training |
| Tool definitions go | into the system prompt | into the request's `tools` field (`schemas`) |
| The model answers with | JSON in its text: `{"tool": "name", "kwargs": {...}}` | a structured tool call, in the reply's `tool_calls` |
| `parse` | finds the JSON in the text | reads the structured call and decodes its arguments |
| Prompt text added | the tool list and the JSON format | none |
| Switch (demo) | `TINYAGENT_TOOLS=1` | `TINYAGENT_NATIVE_TOOLS=1` (wins over `TINYAGENT_TOOLS`) |

This is the difference the project's two models make visible: a baseline model has to be prompted into calling tools, while an agentic one already knows how, so the harness can leave the prompt alone.

## What the agent does with them

`TinyAgent` adds the tool prompt to memory as a system message when it is created. On each task it sends the memory (and `schemas`, for learned tools) to the model, then asks `tools.parse` for a tool call. No tool call means the reply is the answer (`is_done`). A tool call is run by `_execute_action`, and the result is stored as `OBSERVATION: <result>` and recorded in the trajectory.

```{note}
There is no loop yet. After a tool runs, the observation is stored but the model is not called again in the same `run`; the next `run` sees it in memory. The multi-step loop is the next phase.
```

## Adding a tool

```python
from tiny_agent.components.tools.prompting import PromptingTools

tools = PromptingTools(requires_approval=["delete_file"])
tools.add_tool("multiply", lambda a, b: int(a) * int(b), "Multiplies two numbers: multiply(a: str, b: str)")
```

With prompting tools the model writes its arguments as text, so the function converts its own arguments. `default_tools()` in `tiny_agent/components/tools/default.py` registers the built-in `multiply`. The tool functions live one per file in `tiny_agent/components/tools/functions/` (`multiply.py`, `divide.py`); add a new tool as a new file there.

With learned tools, `LearnedTools.schemas` builds the definition from the function itself: a parameter annotated `float` is advertised as `number` (`int` as `integer`, `bool` as `boolean`, anything else as `string`), and its description is the matching entry in the docstring's `Args:` section. So `multiply(a: float | str, b: float | str)` with documented `a` and `b` gives two `number` parameters with descriptions, and a model that sends `6` instead of `"6"` still works because the function calls `float()` on both.

## Complete runs

`tiny_agent/complete/` runs the whole agent against the real model in Docker (`node execute.mjs up llm-client`, then `llm-client load qwen3.5-4b`): `python -m tiny_agent.complete.happy_path` multiplies with prompting tools and then divides that result with learned tools; `python -m tiny_agent.complete.full_run` runs one scenario per part (memory variants, tool approval, settings, the demo, the chat). `node execute.mjs test agent` runs them as `tests/test_complete.py` and skips them when no model is loaded.
