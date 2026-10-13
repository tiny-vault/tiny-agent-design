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

from llm_client.llm import LLM
from tiny_agent.shared.memory import Memory
from tiny_agent.shared.trajectory import Trajectory


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
            A ``Memory`` (``tiny_agent.shared.memory``), passed in when the
            agent is created.
        tools: The "hands": functions the agent can call to affect the world.
            ``None`` until the tools step.
        planner: The "strategy": breaks a goal into ordered sub-steps.
            ``None`` until the planning step.
        trajectory: The :class:`~tiny_agent.shared.trajectory.Trajectory`:
            every run and the steps taken in it, for debugging.

    Example:
        >>> agent = TinyAgent(LLM(model="gemma4-e4b"), Memory())  # doctest: +SKIP
        >>> agent.run("Hello, agent!")  # doctest: +SKIP
        'Hello! How can I help?'
    """

    def __init__(self, llm: LLM, memory: Memory):
        """Assemble the agent's components.

        Role: configuration and dependency wiring. Each attribute is a
        pluggable module, so the agent's capabilities can be swapped or
        extended without changing the control loop.

        Args:
            llm: The client the agent asks for each decision.
            memory: The conversation history the agent reads and writes.
        """
        self.llm = llm
        self.memory = memory
        self.tools = None  # Phase 5: Add Tools
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
            The call is recorded as a :class:`~tiny_agent.shared.step.Step` in
            ``self.trajectory``.
        """
        # The model is stateless, so it gets the full history every time. Storing the
        # reply afterwards is what lets the next call "remember" this one.
        response = self.llm.generate(self.memory.get_messages())
        # A tool-call-only reply has no text (content is None); there is no tool support yet, so treat it as empty text
        text = response.content or ""
        self.memory.add("assistant", text)
        self.trajectory.add(response)
        return text

    def _execute_action(self, action: str) -> str | None:
        """Execute a tool action.

        Role: the "act" phase. Parses the action the LLM chose, dispatches it
        to the matching tool, and returns the result as an observation that is
        stored in memory and fed into the next ``_step``.

        Args:
            action: The action chosen by the model.

        Returns:
            Currently a placeholder, ``"Executed action: <action>"``.
        """
        # Placeholder - will be implemented in later steps
        return f"Executed action: {action}"
