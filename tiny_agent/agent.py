"""The TinyAgent harness: the code that turns an LLM into an agent.

What is it? A language model only turns text into text. An *agent* is that model
plus the code around it (the *harness*): something that keeps the conversation,
decides what to send, and (in later phases) runs tools and plans. ``TinyAgent``
is that harness, built up one phase at a time.

What this file does today: ``run`` puts the task into ``memory``, ``_step`` sends
the whole memory to the model once and stores the reply, and the call is
recorded in ``trajectory``. Because the history is resent every time, a second
``run`` on the same agent can use what the first one said; with a fresh
``Memory`` it cannot. There is no loop, no tool use and no planning yet.

Based on *An Illustrated Guide to AI Agents* by Maarten Grootendorst and
Jay Alammar (O'Reilly). The book's companion repository has no license, so
all rights remain with the authors.
"""

from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import BaseMemory
from tiny_agent.components.tools.tools import BaseTools
from tiny_agent.components.trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework.

    An agent is a loop around an LLM: it perceives a task, decides what to do
    (reason/plan), acts through tools, observes the result, and repeats until
    the task is done::

        task -> [memory] -> LLM decides -> action -> tool -> observation -> (loop)

    The LLM only maps text to text. Everything else that makes it an agent is
    the *harness*, i.e. the code around the model. In this class the harness
    is:

    * :meth:`run` - the loop that keeps calling the model until done.
    * ``_step`` - assembles the model's input (task, memory and tool
      descriptions) and interprets its output.
    * ``_execute_action`` - carries out the model's chosen action in the
      real world and feeds the result back.
    * the attributes below - pluggable slots for memory, tools and planning.

    The model decides; the harness does. The agent now calls the model once
    per task, remembers the exchange in ``memory`` and records what happened
    in ``trajectory``; the loop, tools and planning are still placeholders.

    Memory vs trajectory: ``memory`` is what the model *sees* (the messages sent
    on every call); ``trajectory`` is what *you* see afterwards (a debugging
    record of each run). They hold similar facts for different readers.

    Attributes:
        llm: The "brain": reasons about the task and chooses the next step.
            Passed in when the agent is created.
        memory: The "context": remembers past messages and observations.
            A ``Memory`` (``tiny_agent.components.memory``), passed in when the
            agent is created.
        tools: The "hands": functions the agent can call to affect the world.
            ``None`` until the tools step.
        planner: The "strategy": breaks a goal into ordered sub-steps.
            ``None`` until the planning step.
        trajectory: The :class:`~tiny_agent.components.trajectory.Trajectory`:
            every run and the steps taken in it, for debugging.

    Example:
        >>> agent = TinyAgent(LLM(model="gemma4-e4b"), Memory())  # doctest: +SKIP
        >>> agent.run("Hello, agent!")  # doctest: +SKIP
        'Hello! How can I help?'
    """

    def __init__(self, llm: LLM, memory: BaseMemory, tools: BaseTools | None = None):
        """Assemble the agent's components.

        Role: configuration and dependency wiring. Each attribute is a
        pluggable module, so the agent's capabilities can be swapped or
        extended without changing the control loop.

        Args:
            llm: The client the agent asks for each decision.
            memory: The conversation history the agent reads and writes.
            tools: The tool registry. When given, its prompt is added to memory
                as a system message so the model knows the tools from the start.
                ``None`` means a plain chat agent.
        """
        self.llm = llm
        self.memory = memory
        self.tools = tools
        if tools is not None:
            self.memory.add("system", "You are a helpful assistant.\n\n" + tools.prompt)
        self.planner = None  # Phase 6: Add Planning

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """Run the agent on a task.

        Role: public entry point and outer control loop. Takes a goal from the
        user and drives the agent until it produces a final answer. Later this
        becomes the loop step -> act -> observe -> repeat, with a stop
        condition (answer found, step limit reached, or error).

        Args:
            task: The goal or question for the agent, in natural language.

        Returns:
            The agent's result. Currently the model's reply to the task.
        """
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step()

    def _step(self) -> str:
        """Perform a single step.

        Role: one iteration of the reason-act cycle (the "think" phase).
        Sends the whole memory to the LLM, stores its reply back in memory, and
        returns either a final answer or (later) an action for
        ``_execute_action`` to carry out.

        The task is not passed in: :meth:`run` has already put it into memory,
        which is what the model sees.

        Returns:
            Currently the model's reply text; a reply with no text (a tool call
            only, ``content`` is ``None``) comes back, and is stored, as ``""``.
            The call is recorded as a :class:`~tiny_agent.components.step.Step` in
            ``self.trajectory``.
        """
        # The model is stateless, so it gets the full history every time. Storing the
        # reply afterwards is what lets the next call "remember" this one.
        response = self.llm.generate(
            self.memory.get_messages(),
            tools=self.tools.schemas if self.tools else None,
        )
        # A tool-call-only reply has no text (content is None); treat it as empty text
        text = response.content or ""
        self.memory.add("assistant", text)

        # Without tools every reply is the answer
        if self.tools is None:
            self.trajectory.add(response)
            return text

        # The model asks for a tool by writing JSON in its reply; parse it out of the text
        response = self.tools.parse(response)
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return text
        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """Execute a tool action.

        Role: the "act" phase. Dispatches the parsed tool call the LLM chose to
        the matching tool, and returns the result as an observation that is
        stored in memory and fed into the next ``_step``.

        Args:
            response: The model's reply, with its ``tool_call`` already parsed.

        Returns:
            The observation text, ``"OBSERVATION: <result>"``. It is also added to
            memory and recorded in ``self.trajectory``.
        """
        assert self.tools is not None, "_execute_action needs tools"
        # ACTION: run the tool
        result = self.tools.execute(response)

        # OBSERVATION: store the result so the model can read it, and record the step
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
        return observation
