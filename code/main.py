"""Entry point for the TinyAgent demo: two tasks on one agent, to show its memory.

Role in the agentic system: the "user interface" layer. It creates the agent,
hands it two tasks, and shows the result. It contains no agent logic itself, so
the agent can later be driven from a CLI, a chat loop, or a test just by
changing this file.

Harness: `TinyAgent` (in code/tiny_agent/shared/tiny_agent.py) is the harness, the non-model code that
turns an LLM into an agent. This file only starts it: `main()` creates the
harness and calls `run(task)`, which is where the loop, memory, tools and
planning are meant to live as the phases add them (today `run` makes one
model call, with the conversation so far from `memory`). Keeping the entry point separate means the harness can be
reused unchanged by any front end.

Settings come from ``code/.env`` or the environment: ``TINYAGENT_MODEL``
(default ``gemma4-e4b``), ``TINYAGENT_LLM_URL`` (default the Docker gateway) and
an optional API key. Start the stack and load the model first
(``node execute.mjs up``); the reply varies from run to run.

The demo tells the agent the authors' names, then asks for them in a second run. The second call
sends the whole conversation, so the model can answer; without memory it could not.

Run it with ``python code/main.py``. It logs to ``main.log`` and prints the second answer.
"""

import os
from typing import Any

from cli import api_key_from_env
from llm_client import check, env
from llm_client.llm import LLM
from tiny_agent import log as logs
from tiny_agent.shared.memory import Memory
from tiny_agent.shared.tiny_agent import TinyAgent

# The first task puts the names into memory; the second can only be answered from it.
GREETING = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."
QUESTION = "Hi! What are our names?"


def main() -> None:
    """Build the agent, run the greeting then the question, and print the second answer.

    Role: composition root. Constructs the agent (and, in later steps,
    configures its LLM, memory, tools and planner) before calling `run`.
    """
    env.load_env()  # code/.env: settings such as TINYAGENT_LOG_DIR and TINYAGENT_LOG_LEVEL
    log = logs.setup("main")  # writes to logs/main.log (see code/tiny_agent/log.py)
    # The same settings as the tiny-agent CLI: TINYAGENT_MODEL, TINYAGENT_LLM_URL, TINYAGENT_API_KEY(_FILE).
    api_key = api_key_from_env()
    key_setting: dict[str, Any] = {"api_key": api_key} if api_key else {}
    llm = LLM(
        model=os.environ.get("TINYAGENT_MODEL", "gemma4-e4b"),
        base_url=os.environ.get("TINYAGENT_LLM_URL", check.DEFAULT_URL),
        **key_setting,
    )
    agent = TinyAgent(llm, Memory())
    for task in (GREETING, QUESTION):
        log.info("running task: %s", task)
        result = agent.run(task)
        log.info("result: %s", result)
    print(result)


if __name__ == "__main__":
    main()
