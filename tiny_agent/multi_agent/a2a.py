"""A2A (Agent2Agent): agents that talk to each other over HTTP instead of by calling a Python function.

In ``orchestrator.py`` the specialists are in the same process: ``ask_math_agent`` calls ``math_agent.run``. That works
only when one program owns every agent. A2A is the open protocol that removes the restriction: any agent, written in any
framework and running anywhere, publishes a description of itself and accepts tasks over HTTP, so a client needs nothing
but its URL. The protocol itself comes from the official SDK (``pip install "a2a-sdk[fastapi]" uvicorn``); this file is
the thin layer that connects it to a ``TinyAgent``:

* **Agent card**: a JSON document the server publishes at ``/.well-known/agent-card.json`` saying who the agent is
  (``name``, ``description``, ``skills``) and where it listens. It is how a client discovers what an agent is good for.
* **Task**: a client sends a message (``SendMessage``, JSON-RPC over HTTP); the server answers with a task whose
  ``status.state`` is ``completed`` (the answer is in its ``artifacts``) or ``failed`` (the reason is in
  ``status.message``).

``A2AServer(agent, name, description)`` wraps any ``TinyAgent``: its ``AgentExecutor`` (the SDK's hook for "what to do
with a message") calls ``agent.run`` and publishes the answer as an artifact. The agent keeps its own memory between
messages, as a specialist does in-process. ``A2AClient(url)`` is the other side: ``send(text)`` returns the answer text.
It knows nothing of ``TinyAgent`` and works against any A2A server.

The SDK is asynchronous and ``TinyAgent`` is not, so the server runs ``uvicorn`` in a background thread and executes
``agent.run`` in a worker thread (one task at a time, because an agent is not thread-safe), and ``send`` blocks on
``asyncio.run``, which cannot be called from inside a running event loop (a notebook, for example). Left out on
purpose: streaming, authentication, push notifications, tasks that wait for more input. It binds ``127.0.0.1`` and
trusts every caller.

Starting a server binds a port and a thread (``port=0`` lets the system choose a free one, read it from ``url``);
``stop`` releases both. Compare ``orchestrator.py`` (the same two specialists called in-process) and run
``python -m tiny_agent.multi_agent.orchestrator --a2a`` for the protocol version.
"""

import asyncio
import socket
import threading
import time
from typing import override

import httpx
import uvicorn
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import (
    get_artifact_text,
    get_message_text,
    new_task_from_user_message,
    new_text_artifact,
    new_text_message,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
    SendMessageRequest,
    TaskState,
)
from starlette.applications import Starlette

from tiny_agent.agent import TinyAgent


class TinyAgentExecutor(AgentExecutor):
    """Connects the SDK to a ``TinyAgent``: a message in, ``agent.run``, an artifact and a final state out."""

    def __init__(self, agent: TinyAgent):
        self.agent = agent
        self._one_at_a_time = asyncio.Lock()  # the agent's memory and trajectory are shared state

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run the agent on the message text; a raised exception becomes a ``failed`` task, not a crashed server."""
        message = context.message
        if message is None:
            return
        task = context.current_task
        if task is None:  # the first event of a new task is the task itself
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            async with self._one_at_a_time:
                answer = await asyncio.to_thread(self.agent.run, get_message_text(message))
        except Exception as error:  # noqa: BLE001 - any failure of the agent is reported on the task
            await updater.failed(new_text_message(f"{type(error).__name__}: {error}"))
            return
        await updater.add_artifact(list(new_text_artifact("answer", answer).parts), name="answer")
        await updater.complete()

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Not supported: ``TinyAgent.run`` cannot be interrupted."""


class A2AServer:
    """Serve one ``TinyAgent`` over A2A, in a background thread.

    Attributes:
        agent: The agent that answers every message.
        card: The agent card served at ``/.well-known/agent-card.json``.
    """

    def __init__(self, agent: TinyAgent, name: str, description: str, host: str = "127.0.0.1", port: int = 0):
        """Bind the port and build the app; nothing is served until :meth:`start`.

        Args:
            agent: The agent to expose.
            name: The card's ``name``.
            description: The card's ``description`` and its single skill's description: what to ask this agent.
            host: The interface to bind. Keep the loopback address; there is no authentication.
            port: The port; ``0`` picks a free one.
        """
        self.agent = agent
        self._socket = socket.socket()  # bound now, so the card can carry the real URL before serving starts
        self._socket.bind((host, port))
        self.url = f"http://{host}:{self._socket.getsockname()[1]}"
        self.card = AgentCard(
            name=name,
            description=description,
            version="1.0.0",
            supported_interfaces=[AgentInterface(url=self.url, protocol_binding="JSONRPC", protocol_version="1.0")],
            capabilities=AgentCapabilities(streaming=False),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[AgentSkill(id="ask", name=name, description=description, tags=[])],
        )
        handler = DefaultRequestHandler(
            agent_executor=TinyAgentExecutor(agent), task_store=InMemoryTaskStore(), agent_card=self.card
        )
        app = Starlette(routes=create_agent_card_routes(self.card) + create_jsonrpc_routes(handler, "/"))
        self._server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self._thread: threading.Thread | None = None

    def start(self) -> "A2AServer":
        """Start answering in a daemon thread (it ends with the program), wait until it is ready, return ``self``."""
        self._thread = threading.Thread(target=lambda: asyncio.run(self._server.serve(sockets=[self._socket])), daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"The A2A server on {self.url} did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        """Stop answering and release the port."""
        if self._thread is not None:
            self._server.should_exit = True
            self._thread.join()
            self._thread = None
        self._socket.close()


class A2AClient:
    """Talk to an A2A agent by its URL. Knows nothing about ``TinyAgent``: any A2A server works."""

    def __init__(self, url: str, timeout: float = 600.0):
        """
        Args:
            url: The agent's base URL, for example ``http://127.0.0.1:50321``.
            timeout: Seconds to wait for an answer. Long, because the remote agent runs a whole model loop.
        """
        self.url = url.rstrip("/")
        self.timeout = timeout

    def card(self) -> AgentCard:
        """Fetch the agent card (what the agent is and what it can do).

        Raises:
            RuntimeError: The agent cannot be reached or does not publish a card.
        """

        async def fetch() -> AgentCard:
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                return await A2ACardResolver(http, self.url).get_agent_card()

        try:
            return asyncio.run(fetch())
        except Exception as error:
            raise RuntimeError(f"Cannot read the agent card at {self.url}: {error}") from error

    def send(self, text: str) -> str:
        """Send one text message and return the agent's answer text.

        Args:
            text: The question or task, in natural language.

        Raises:
            RuntimeError: The agent cannot be reached, or its task did not complete (the message carries the agent's
                own reason).
        """
        try:
            return asyncio.run(self._send(text))
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"Cannot reach the agent at {self.url}: {error}") from error

    async def _send(self, text: str) -> str:
        """The SDK call: stream the (single) reply and read the answer from the finished task or message."""
        # The SDK's default HTTP client gives up after 5 s, far less than a model loop takes, so pass our own timeout.
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            client = await create_client(self.url, ClientConfig(streaming=False, httpx_client=http))
            request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))
            answer = ""
            try:
                async for response in client.send_message(request):
                    if response.HasField("message"):
                        answer = get_message_text(response.message)
                    elif response.HasField("task"):
                        task = response.task
                        if task.status.state != TaskState.TASK_STATE_COMPLETED:
                            reason = get_message_text(task.status.message) if task.status.HasField("message") else ""
                            state = TaskState.Name(task.status.state)
                            raise RuntimeError(f"The agent at {self.url} ended in {state}: {reason}")
                        answer = "\n".join(get_artifact_text(artifact) for artifact in task.artifacts)
            finally:
                await client.close()
            return answer
