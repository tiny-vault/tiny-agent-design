"""Send one question to a model through the LLM client and print the answer.

This is the llm-client container's way to really talk to a model (``python -m llm_client.check``, its
default command, only lists which models are loaded). It uses ``llm_client.llm.LLM``, the same client the
TinyAgent uses, against the same server the chat app talks to: ``TINYAGENT_LLM_URL``, which defaults to
the chat app's gateway on this machine. So the model that answers is the model loaded in the chat app.

Usage (inside the container; ``node execute.mjs llm-client ask "..."`` runs it for you)::

    python /app/ask.py "Why is the sky blue?"
    python /app/ask.py --model qwen3.5-4b "Why is the sky blue?"

Without ``--model`` it uses the one model that is loaded right now. It never loads a model itself: loading
takes tens of seconds and unloads the other one, so that stays an explicit step
(``node execute.mjs load <model>``).

Exit codes: ``0`` answered, ``1`` the model call failed, ``2`` wrong usage or no model server / no model
loaded to ask.
"""

import os
import sys

from llm_client import check
from llm_client.llm import LLM

USAGE = 'usage: python ask.py [--model NAME] "question"'


def parse(argv: list[str]) -> tuple[str | None, str]:
    """Split the arguments into the optional model name and the question.

    Args:
        argv: The arguments after the script name.

    Returns:
        ``(model, question)``; ``model`` is ``None`` when ``--model`` was not given. The question is every
        remaining word joined with spaces, so quoting it is optional.
    """
    args = list(argv)
    model = None
    if "--model" in args:
        i = args.index("--model")
        if i + 1 >= len(args):
            return None, ""
        model = args[i + 1]
        del args[i : i + 2]
    return model, " ".join(args).strip()


def main(argv: list[str]) -> int:
    """Ask the question and print the answer. Returns the process exit code."""
    model, question = parse(argv)
    if not question:
        print(USAGE)
        return 2

    base_url = os.environ.get("TINYAGENT_LLM_URL", check.DEFAULT_URL)
    statuses = check.model_statuses(base_url)
    if statuses is None:
        print(f"Cannot reach the model server at {base_url}. Is it running? (node execute.mjs up llm-client)")
        return 2

    # The chat app keeps one model loaded; asking that one is what "the same model as the chat app" means.
    if model is None:
        loaded = [name for name, status in statuses.items() if status == "loaded"]
        if not loaded:
            print("No model is loaded. Load one with: node execute.mjs llm-client load <model>")
            return 2
        model = loaded[0]

    try:
        response = LLM(model=model, base_url=base_url).generate([{"role": "user", "content": question}])
    except RuntimeError as error:  # LLM.generate turns every failure into one RuntimeError with the fix in it
        print(error)
        return 1
    print(f"[{model}]")
    print(response.content or "(the model sent no text)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
