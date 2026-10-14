"""The complete run of TinyAgent, every part of it, against the real Docker model servers.

What is it? One scenario per part of the harness, each a normal successful use with real model requests: the chat model on
``TINYAGENT_LLM_URL`` (default ``http://127.0.0.1:3200/v1``) and EmbeddingGemma on ``TINYAGENT_EMBEDDING_URL``
(default ``http://127.0.0.1:3201/v1``). Start them with ``node execute.mjs up llm-client`` and load a chat model with
``node execute.mjs llm-client load qwen3.5-4b``. ``happy_path.py`` next to this file is the shorter run: one tool by
prompting, then one by learned calling.

The scenarios, in order:

* ``server_check``: which models the server has loaded (``check``, ``pick_model``, the ``check`` command).
* ``embedding``: ``EmbeddingModel`` and ``cosine``; related texts score higher than unrelated ones.
* ``plain_chat``: ``Memory``: the second task is answered from the first. Trajectory and token counts.
* ``thinking``: ``LLM(think=True)`` returns the model's reasoning.
* ``prompting_tools`` / ``native_tools``: a tool call, its observation and the answer, both ways of telling the model.
* ``tool_approval``: ``requires_approval``: a "no" is reported to the model, a "yes" runs the tool.
* ``trimming_memory``, ``summarization_memory``, ``rag_memory``, ``memory_bank``: each memory variant.
* ``settings_wiring``: ``Settings`` flags choose the memory and the tools (``memory_from_env``, ``tools_from_settings``).
* ``demo_main``: ``python -m tiny_agent.main`` end to end, including its ``main.log``.
* ``cli_chat``: a scripted terminal chat session with the slash commands.

Chat replies differ from run to run, so a scenario checks for the facts that must be there (a name, ``42``, a tool
call) and never for exact text. A failed check raises ``AssertionError`` naming what was missing.

Scope and limits: local environment and approval input are controlled with
``patch``; this is not a fully unmodified user session. Scenarios send model
requests and can write temporary logs. A passing arithmetic or retrieval
example does not establish general model reliability. ``thinking`` reports
returned reasoning length but does not require nonempty reasoning. The bank
uses a movable clock to demonstrate its decay rule, not real elapsed days.
Unknown scenario names raise ``KeyError``; only failed checks and missing
servers have explicit handling here.

Run it with ``python -m tiny_agent.complete.full_run`` (all scenarios) or ``... full_run tool_approval rag_memory``.
"""

import contextlib
import io
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tiny_agent import cli
from tiny_agent import main as demo
from tiny_agent.agent import TinyAgent
from tiny_agent.complete.happy_path import StackNotRunning, real_llm, require
from tiny_agent.components.client import check
from tiny_agent.components.client.embedding import EmbeddingModel
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import BaseMemory, Memory
from tiny_agent.components.memory.memory_bank import MemoryBank
from tiny_agent.components.memory.rag import RAGMemory
from tiny_agent.components.memory.similarity import cosine
from tiny_agent.components.memory.summarization import SummarizationMemory
from tiny_agent.components.memory.trimming import TrimmingMemory
from tiny_agent.components.tools.default import (
    default_native_tools,
    default_tools,
    tools_from_settings,
)
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.components.tools.prompting import PromptingTools
from tiny_agent.config.settings import Settings

Facts = dict[str, Any]

MULTIPLY_TASK = 'Use the multiply tool with a="6" and b="7".'
ANSWER_PROMPT = "In one short sentence, what was the result?"
# Settings that say where the stack is; every other TINYAGENT_* variable is cleared for a scenario.
KEEP = ("TINYAGENT_LLM_URL", "TINYAGENT_EMBEDDING_URL", "TINYAGENT_MODEL", "TINYAGENT_API_KEY", "TINYAGENT_API_KEY_FILE")


def contains(text: str | None, word: str) -> bool:
    """Whether ``word`` is in ``text``, in any case."""
    return word.lower() in (text or "").lower()


def embedder(llm: LLM) -> EmbeddingModel:
    """A client for the embedding model.

    Raises:
        StackNotRunning: The embedding server is down.
    """
    settings = Settings.from_env()
    model = EmbeddingModel(settings.embedding_model, base_url=settings.embedding_url, api_key=llm.api_key)
    try:
        model.embed("ping")
    except RuntimeError as error:
        raise StackNotRunning(
            f"no embedding model at {settings.embedding_url} ({error}); run: node execute.mjs up llm-client"
        ) from error
    return model


@contextlib.contextmanager
def clean_env(**values: str) -> Iterator[Path]:
    """A scratch folder, and an environment with no ``TINYAGENT_*`` settings except where the stack is and ``values``.

    The ``.env`` file is switched off and the logs go to the scratch folder, so a scenario neither reads nor writes
    the developer's own files.
    """
    kept = {k: v for k, v in os.environ.items() if not k.startswith("TINYAGENT_") or k in KEEP}
    with tempfile.TemporaryDirectory() as folder:
        kept["TINYAGENT_ENV_FILE"] = str(Path(folder) / "none.env")
        kept["TINYAGENT_LOG_DIR"] = str(Path(folder) / "logs")
        kept.update(values)
        with patch.dict(os.environ, kept, clear=True):
            yield Path(folder)


def tell_names(agent: TinyAgent) -> str:
    """Tell the agent the authors' names, then ask for them; return the second answer."""
    agent.run(demo.GREETING)
    return agent.run(demo.QUESTION)


# ---- the model client --------------------------------------------------------------------------------------


def server_check() -> Facts:
    """The helpers that ask the server what is loaded."""
    url = Settings.from_env().llm_url
    llm = real_llm()
    statuses = check.model_statuses(url)
    require(statuses and statuses[llm.model] == "loaded", f"{llm.model} should be loaded: {statuses}")
    require(llm.model in check.loaded_models(url), "loaded_models should list the loaded model")
    require(check.model_status(llm.model, url) == "loaded", "model_status should say loaded")
    require(check.model_status("no-such-model", url) is None, "an unknown model has no status")
    require(cli.pick_model(llm.model, url) == (llm.model, ""), "pick_model should take the loaded model as asked")
    require(cli.pick_model(None, url)[0] in check.loaded_models(url), "pick_model should default to a loaded model")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = check.main([llm.model, "--url", url])
    require(code == 0, f"the check command should exit 0 for a loaded model, got {code}: {out.getvalue()}")
    return {"model": llm.model, "statuses": statuses}


def embedding() -> Facts:
    """EmbeddingGemma turns texts into vectors; related texts are closer."""
    model = embedder(real_llm())
    vector = model.embed("Sarah lives in Lisbon.")
    related = cosine(vector, model.embed("Which city does Sarah live in?"))
    unrelated = cosine(vector, model.embed("Ilse plays the cello."))
    require(len(vector) > 100, f"an embedding has many dimensions, got {len(vector)}")
    require(related > unrelated, f"related texts should be closer: {related:.3f} vs {unrelated:.3f}")
    return {"dimensions": len(vector), "related": round(related, 3), "unrelated": round(unrelated, 3)}


def plain_chat() -> Facts:
    """Two tasks on one agent with ``Memory``: the second is answered from the first."""
    agent = TinyAgent(real_llm(), Memory())
    answer = tell_names(agent)
    require(contains(answer, "Maarten") and contains(answer, "Jay"), f"the answer should hold both names: {answer!r}")
    require(len(agent.memory.get_messages()) == 4, "memory should hold user, assistant, user, assistant")
    first, second = agent.trajectory.runs
    require((first["query"], second["query"]) == (demo.GREETING, demo.QUESTION), "the trajectory records both tasks")
    step = second["steps"][0]
    require(step.answer == answer and step.action is None, "a plain reply is recorded as the answer")
    require(step.metadata and step.metadata["completion_tokens"], "the step keeps the token counts")
    return {"answer": answer, "model": step.metadata["model"], "completion_tokens": step.metadata["completion_tokens"]}


def thinking() -> Facts:
    """With ``think=True`` the model may reason first; the reasoning is recorded as the step's ``thought``."""
    llm = real_llm()
    llm.think = True
    agent = TinyAgent(llm, Memory())
    answer = agent.run("What is 17 + 25? Answer with the number.")
    (step,) = agent.trajectory.runs[0]["steps"]
    require(contains(answer, "42"), f"17 + 25 is 42: {answer!r}")
    return {"answer": answer, "thought_chars": len(step.thought)}


# ---- tools -------------------------------------------------------------------------------------------------


def tool_round_trip(agent: TinyAgent) -> Facts:
    """The shared part of the tool scenarios: tool call, observation, then the answer from the observation."""
    observation = agent.run(MULTIPLY_TASK)
    require(observation == "OBSERVATION: 42.0", f"the harness should run multiply and return its result: {observation!r}")
    (step,) = agent.trajectory.runs[0]["steps"]
    require(step.action and step.action["tool"] == "multiply", f"the step should record the tool call: {step.action}")
    require(step.observation == observation and step.answer is None, "the step holds the observation, not an answer")
    require(agent.memory.get_messages()[-1] == {"role": "user", "content": observation}, "the observation goes to memory")
    answer = agent.run(ANSWER_PROMPT)
    require(contains(answer, "42"), f"the model should read the observation and answer 42: {answer!r}")
    return {"action": step.action, "observation": observation, "answer": answer}


def prompting_tools() -> Facts:
    """The model is told about the tools in the system prompt and replies with JSON."""
    tools = default_tools()
    agent = TinyAgent(real_llm(), Memory(), tools)
    first = agent.memory.get_messages()[0]
    require(first["role"] == "system" and tools.prompt in first["content"], "the tool prompt is the first message")
    require(tools.schemas is None, "prompting tools send no `tools` field")
    return tool_round_trip(agent)


def native_tools() -> Facts:
    """The model learned tool calling: the definitions travel in the request's ``tools`` field."""
    tools = default_native_tools()
    schemas = tools.schemas
    require(schemas[0]["function"]["name"] == "multiply", "the schema names the tool")
    require(schemas[0]["function"]["parameters"]["required"] == ["a", "b"], "both parameters are required")
    require(tools.prompt == "", "learned tools add no tool text to the prompt")
    return {"schema": schemas[0]["function"]["name"], **tool_round_trip(TinyAgent(real_llm(), Memory(), tools))}


def tool_approval() -> Facts:
    """A tool in ``requires_approval`` asks first: "n" is told to the model, "y" runs it."""
    outcomes = {}
    for answer, expected in (("n", "OBSERVATION: Tool 'multiply' was denied by the user."), ("y", "OBSERVATION: 42.0")):
        agent = TinyAgent(real_llm(), Memory(), default_tools(requires_approval=["multiply"]))
        with patch("builtins.input", lambda prompt, answer=answer: answer):
            outcomes[answer] = agent.run(MULTIPLY_TASK)
        require(outcomes[answer] == expected, f"answering {answer!r} should give {expected!r}, got {outcomes[answer]!r}")
    return outcomes


# ---- memory ------------------------------------------------------------------------------------------------


def trimming_memory() -> Facts:
    """``TrimmingMemory`` keeps only the last two turns, so the first fact is gone after a few tasks."""
    agent = TinyAgent(real_llm(), TrimmingMemory())
    agent.run(demo.GREETING)
    for task in ("What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?"):
        agent.run(task)
    messages = agent.memory.get_messages()
    require(len(messages) == 4, f"only the last two turns are kept, got {len(messages)} messages")
    require(not any(contains(m["content"], "Maarten") for m in messages), "the greeting should have left the window")
    return {"kept": len(messages)}


def summarization_memory() -> Facts:
    """``SummarizationMemory`` replaces the history with a model-written summary that still holds the names."""
    llm = real_llm()
    agent = TinyAgent(llm, SummarizationMemory(llm))
    answer = tell_names(agent)
    (summary,) = agent.memory.get_messages()
    require(summary["role"] == "system", "the history is one summary message")
    require(contains(summary["content"], "Maarten"), f"the summary should keep the names: {summary['content']!r}")
    require(contains(answer, "Maarten"), f"the model should answer from the summary: {answer!r}")
    return {"answer": answer, "summary": summary["content"]}


def rag_memory() -> Facts:
    """``RAGMemory`` puts the documents that match the question in front of it."""
    llm = real_llm()
    memory = RAGMemory(embedder(llm), demo.DOCUMENTS)
    require(len(memory.embeddings) == len(demo.DOCUMENTS), "every document is embedded once, at construction")
    require(memory.search("Where does Sarah live?")[0] == "Sarah lives in Lisbon, Portugal.", "retrieval finds Sarah's city")
    answer = TinyAgent(llm, memory).run("Where does Sarah live? Answer with the city.")
    stored = memory.get_messages()[0]["content"]
    require(stored.startswith("Context:") and "Lisbon" in stored, "the user message is stored with its context")
    require(contains(answer, "Lisbon"), f"the model should answer from the context: {answer!r}")
    return {"answer": answer}


class Clock:
    """A clock the scenario can move forward, so forgetting does not need days of waiting."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def days(self, count: float) -> None:
        self.now += count * 86400


def memory_bank() -> Facts:
    """``MemoryBank`` summarises each turn, keeps a user portrait, strengthens what it recalls and forgets the rest."""
    llm = real_llm()
    clock = Clock()
    bank = MemoryBank(llm, embedder(llm), clock=clock)
    agent = TinyAgent(llm, bank)
    agent.run("Hi, I am Maarten and I play the cello in a quartet.")
    require(len(bank.event_summaries) == 1 and bank.event_summaries[0]["strength"] == 1, "a turn becomes one summary")
    require(bank.user_portrait, "the portrait is written after the first reply")
    clock.days(1)
    answer = agent.run("Which instrument do I play?")
    require(bank.event_summaries[0]["strength"] == 2, "recalling a summary makes it stronger")
    recalled = bank.get_messages()[2]["content"]
    require("Memories:" in recalled and contains(recalled, "cello"), "the recalled summary is put in front of the question")
    require(len(bank.conversation_history) == 4, "the raw conversation is kept, timestamped")
    clock.days(500)
    require(bank.search("Which instrument do I play?") == [], "after a long time every summary has faded")
    return {"answer": answer, "summaries": len(bank.event_summaries), "portrait": bank.user_portrait}


# ---- configuration and front ends --------------------------------------------------------------------------


def settings_wiring() -> Facts:
    """The settings flags choose the memory and the tools, and the strongest memory flag wins."""
    llm = real_llm()
    cases: dict[str, tuple[dict[str, bool], type[BaseMemory]]] = {
        "plain": ({}, Memory),
        "trim": ({"trim_memory": True}, TrimmingMemory),
        "summary": ({"summary_memory": True, "trim_memory": True}, SummarizationMemory),
        "rag": ({"rag_memory": True, "summary_memory": True}, RAGMemory),
        "bank": ({"bank_memory": True, "rag_memory": True}, MemoryBank),
    }
    chosen: Facts = {}
    with clean_env() as folder:
        for kind, (flags, expected) in cases.items():
            settings = Settings(**flags)  # pyright: ignore[reportArgumentType]
            require(settings.memory == kind, f"{flags} should choose {kind}, got {settings.memory}")
            memory = demo.memory_from_env(llm, settings)
            require(type(memory) is expected, f"{kind} should build {expected.__name__}, got {type(memory).__name__}")
            chosen[kind] = type(memory).__name__
        require(tools_from_settings(Settings()) is None, "no flag, no tools")
        require(type(tools_from_settings(Settings(tools=True))) is PromptingTools, "TINYAGENT_TOOLS binds prompting tools")
        require(type(tools_from_settings(Settings(native_tools=True, tools=True))) is LearnedTools, "native tools win")
        key_file = folder / "key"
        key_file.write_text("# the key\nsecret-key\n", encoding="utf-8")
        require(Settings(api_key_file=key_file).resolve_api_key() == "secret-key", "the key file's first real line is the key")
    return chosen


def demo_main() -> Facts:
    """``main()``: build the agent from the settings, run both tasks, print the answer, write ``main.log``."""
    model = real_llm().model  # main() would default to gemma4-e4b, which may not be the one loaded
    with clean_env(TINYAGENT_MODEL=model) as folder, contextlib.redirect_stdout(io.StringIO()) as out:
        logger = logging.getLogger("tiny_agent.main")
        try:
            demo.main()
            log = (folder / "logs" / "main.log").read_text(encoding="utf-8")
        finally:
            # main() leaves its log file open; Windows cannot delete the scratch folder until it is closed
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    printed = out.getvalue()
    require(contains(printed, "Maarten"), f"main should print the answer with the names: {printed!r}")
    require("running task:" in log and "result:" in log, "main.log records each task and its result")
    return {"printed": printed.strip()}


def cli_chat() -> Facts:
    """A terminal chat session: slash commands, two questions, ``/clear``, then ``/exit``."""
    llm = real_llm()
    lines = iter(
        ["/help", "/models", "/model", "/think on", "/think off", "My name is Maarten.", "What is my name?", "/clear", "/exit"]
    )
    output: list[str] = []
    cli.chat(llm, system="Answer in one short sentence.", input_fn=lambda prompt: next(lines), output=output.append)
    shown = "\n".join(output)
    require(f"* {llm.model}" in shown, "/models marks the current model")
    require("Thinking is on." in shown and "Thinking is off." in shown, "/think switches thinking")
    require("Conversation cleared." in shown, "/clear confirms")
    replies = [line for line in output if line.startswith(f"{llm.model}> ")]
    require(len(replies) == 2, f"the two questions should be answered, got {replies}")
    require(contains(replies[1], "Maarten"), f"the chat remembers the first message: {replies[1]!r}")
    return {"replies": replies}


SCENARIOS: dict[str, Callable[[], Facts]] = {
    scenario.__name__: scenario
    for scenario in (
        server_check, embedding, plain_chat, thinking, prompting_tools, native_tools, tool_approval, trimming_memory,
        summarization_memory, rag_memory, memory_bank, settings_wiring, demo_main, cli_chat,
    )  # fmt: skip
}


def main(argv: list[str] | None = None) -> int:
    """Run the scenarios named on the command line (all by default), print each, and return the exit code."""
    names = (sys.argv[1:] if argv is None else argv) or list(SCENARIOS)
    failed = 0
    for name in names:
        try:
            facts = SCENARIOS[name]()
        except StackNotRunning as error:
            print(f"cannot run: {error}")
            return 2
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}: {error}")
            continue
        print(f"ok   {name}")
        for key, value in facts.items():
            print(f"       {key}: {value}")
    print(f"\n{len(names) - failed}/{len(names)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
