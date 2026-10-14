"""Tests for A2A (on the official SDK): the agent card, a message round-trip over a real local socket, failures, and the orchestrator's a2a flag."""

import httpx
import pytest

from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import Memory
from tiny_agent.components.planning.native_react import NativeReAct
from tiny_agent.components.tools.learned import LearnedTools
from tiny_agent.config.settings import Settings
from tiny_agent.multi_agent.a2a import A2AClient, A2AServer
from tiny_agent.multi_agent.orchestrator import create_orchestrator
from tiny_agent.tests.stubs import ScriptedLLM


def specialist(*replies: str) -> TinyAgent:
    return TinyAgent(ScriptedLLM(*replies), Memory(), LearnedTools(), NativeReAct())


@pytest.fixture
def served():
    """A scripted agent behind a real server on a free port; stopped afterwards."""
    server = A2AServer(specialist("42", "43"), "Math agent", "Does sums.").start()
    yield server
    server.stop()


def test_the_agent_card_says_who_the_agent_is_and_where(served: A2AServer):
    card = A2AClient(served.url).card()
    assert (card.name, card.description) == ("Math agent", "Does sums.")
    assert card.supported_interfaces[0].url == served.url and card.skills[0].description == "Does sums."
    assert httpx.get(served.url + "/.well-known/agent-card.json").json()["name"] == "Math agent"


def test_send_returns_the_answer_and_the_agent_keeps_its_memory(served: A2AServer):
    client = A2AClient(served.url)
    assert [client.send("6 x 7?"), client.send("and plus one?")] == ["42", "43"]
    assert [run["query"] for run in served.agent.trajectory.runs] == ["6 x 7?", "and plus one?"]


def test_a_task_that_raises_ends_failed_with_the_reason_and_the_server_keeps_running():
    server = A2AServer(specialist("fine"), "Fragile", "Has one reply.").start()
    client = A2AClient(server.url)
    assert client.send("first") == "fine"
    with pytest.raises(RuntimeError, match="TASK_STATE_FAILED: IndexError"):  # ScriptedLLM has run out
        client.send("second")
    server.stop()


def test_the_client_names_the_url_when_nothing_answers():
    server = A2AServer(specialist(), "Gone", "Stopped.").start()
    url = server.url
    server.stop()
    with pytest.raises(RuntimeError, match="Cannot reach the agent"):
        A2AClient(url, timeout=2).send("hello")


def test_the_flag_is_a_setting_off_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TINYAGENT_A2A", raising=False)
    assert Settings(_env_file=None).a2a is False  # pyright: ignore[reportCallIssue]
    monkeypatch.setenv("TINYAGENT_A2A", "on")
    assert Settings(_env_file=None).a2a is True  # pyright: ignore[reportCallIssue]


def call(name: str, question: str) -> dict[str, object]:
    arguments = '{"question": "' + question + '"}'
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_with_the_flag_the_orchestrator_reaches_the_specialists_over_http():
    math_agent = specialist("4696")
    llm = ScriptedLLM(
        Response(content=None, tool_call=call("ask_math_agent", "What is 1174 times 4?")),
        Response(content="EUR 4696"),
    )
    orchestrator = create_orchestrator(llm, math_agent=math_agent, date_agent=specialist(), a2a=True)
    assert orchestrator.run("save 4 a day") == "EUR 4696"
    # The question arrived at the specialist as an A2A message, and the answer came back as the observation
    assert math_agent.trajectory.runs[0]["query"] == "What is 1174 times 4?"
    assert orchestrator.trajectory.runs[0]["steps"][0].observation == "OBSERVATION: 4696"
