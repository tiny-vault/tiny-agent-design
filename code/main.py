"""Entry point for the TinyAgent demo.

Role in the agentic system: the "user interface" layer. It creates the agent,
hands it a task, and shows the result. It contains no agent logic itself, so
the agent can later be driven from a CLI, a chat loop, or a test just by
changing this file.

Harness: `TinyAgent` (in code/tiny_agent/shared/tiny_agent.py) is the harness, the non-model code that
turns an LLM into an agent. This file only starts it: `main()` creates the
harness and calls `run(task)`, which is where the loop, memory, tools and
planning live. Keeping the entry point separate means the harness can be
reused unchanged by any front end.
"""

from llm_client import env
from tiny_agent import log as logs
from tiny_agent.shared.tiny_agent import TinyAgent


def main() -> None:
    """Build the agent, run one task, and print the answer.

    Role: composition root. Constructs the agent (and, in later steps,
    configures its LLM, memory, tools and planner) before calling `run`.
    """
    env.load_env()  # code/.env: settings such as TINYAGENT_LOG_DIR and TINYAGENT_LOG_LEVEL
    log = logs.setup("main")  # writes to logs/main.log (see code/tiny_agent/log.py)
    agent = TinyAgent()
    log.info("running task: %s", "Hello, agent!")
    result = agent.run("Hello, agent!")
    log.info("result: %s", result)
    print(result)


if __name__ == "__main__":
    main()
