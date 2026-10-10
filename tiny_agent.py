"""The TinyAgent harness: the code that turns an LLM into an agent.

Based on *An Illustrated Guide to AI Agents* by Maarten Grootendorst and
Jay Alammar (O'Reilly). The book's companion repository has no license, so
all rights remain with the authors.
"""


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

    The model decides; the harness does. Right now these are placeholders, so
    the harness is only a skeleton and no model is called yet.

    Attributes:
        llm: The "brain": reasons about the task and chooses the next step.
            ``None`` until the LLM step.
        memory: The "context": remembers past messages and observations.
            ``None`` until the memory step.
        tools: The "hands": functions the agent can call to affect the world.
            ``None`` until the tools step.
        planner: The "strategy": breaks a goal into ordered sub-steps.
            ``None`` until the planning step.

    Example:
        >>> agent = TinyAgent()
        >>> agent.run("Hello, agent!")
        'Received: Hello, agent!'
    """

    def __init__(self):
        """Assemble the agent's components.

        Role: configuration and dependency wiring. Each attribute is a
        pluggable module, so the agent's capabilities can be swapped or
        extended without changing the control loop.
        """
        self.llm = None  # Step 1: Add LLM
        self.memory = None  # Step 2: Add Memory
        self.tools = None  # Step 3: Add Tools
        self.planner = None  # Step 4: Add Planning

    def run(self, task: str) -> str:
        """Run the agent on a task.

        Role: public entry point and outer control loop. Takes a goal from the
        user and drives the agent until it produces a final answer. Later this
        becomes the loop step -> act -> observe -> repeat, with a stop
        condition (answer found, step limit reached, or error).

        Args:
            task: The goal or question for the agent, in natural language.

        Returns:
            The agent's result. Currently ``"Received: <task>"``.
        """
        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single step.

        Role: one iteration of the reason-act cycle (the "think" phase).
        Builds the prompt from the task plus memory, asks the LLM what to do
        next, and returns either a final answer or an action for
        ``_execute_action`` to carry out.

        Args:
            task: The current task text.

        Returns:
            Currently a placeholder echo, ``"Received: <task>"``.
        """
        # Placeholder - will be implemented in later steps
        return f"Received: {task}"

    def _execute_action(self, action: str) -> str:
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
