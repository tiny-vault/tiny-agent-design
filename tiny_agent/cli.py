"""Chat with the model from the terminal.

Install the project once (from the repo root) and you get a ``tiny-agent`` command::

    pip install -e .
    tiny-agent                       # interactive chat
    tiny-agent "Why is the sky blue?"   # one question, one answer, then exit
    tiny-agent --demo                # run main.py's demo first, then chat

It also works without installing: ``python -m tiny_agent.cli``.

The chat talks to the model through :class:`~tiny_agent.llm_client.llm.LLM`, so it
uses the Docker gateway by default (start it with
``node execute.mjs up``). It picks the model that is
loaded right now unless you name one with ``--model``.

Inside the chat, lines starting with ``/`` are commands (``/help`` lists them).
Standard library only.

How the chat works: the model server keeps no memory between requests, so
:func:`chat` keeps a ``messages`` list and sends all of it on every turn. Each turn
appends the user's line, calls ``llm.generate``, and appends the model's reply. That
growing list is what lets the model "remember" the conversation (the agent's
``Memory`` class does the same job). :func:`main` picks a model, builds the ``LLM``
and then either answers one question or starts the chat.

Limitations: no streaming (a reply appears all at once), no history trimming (a very
long chat will exceed the model's context window), and this chat does not use
``TinyAgent``. Replies vary from run to run.
"""

import argparse
import logging
import os
import sys

from tiny_agent import log as logs
from tiny_agent.llm_client import check, env
from tiny_agent.llm_client.llm import LLM

# Written to logs/cli.log once main() has called logs.setup("cli"). The NullHandler
# keeps chat() and pick_model() silent when they are used without main() (in tests).
log = logging.getLogger("tiny_agent.cli")
log.addHandler(logging.NullHandler())

HELP = """Commands:
  /help            show this help
  /models          list the models and which one is loaded
  /model NAME      switch to another loaded model
  /think on|off    let the model reason before answering (slower)
  /clear           forget the conversation so far
  /exit            leave (Ctrl-D and Ctrl-C also work)
Anything else is sent to the model."""


def pick_model(requested: str | None, base_url: str) -> tuple[str | None, str]:
    """Choose the model to chat with.

    Args:
        requested: The model the user asked for, or ``None`` to use whatever is loaded.
        base_url: The API root, ending in ``/v1``.

    Returns:
        ``(model, note)``. ``model`` is ``None`` when no model can be used, and
        ``note`` then says why and what to do. When a model is chosen, ``note``
        is empty or says that a different model was substituted.
    """
    statuses = check.model_statuses(base_url)
    log.debug("model statuses from %s: %s", base_url, statuses)
    if statuses is None:
        return None, f"Cannot reach the model server at {base_url}. Is the stack running? (node execute.mjs up)"
    loaded = [m for m, s in statuses.items() if s == "loaded"]
    router = any(s in ("loaded", "loading", "unloaded") for s in statuses.values())

    if requested:
        status = statuses.get(requested)
        if status in ("loaded", "unknown"):
            return requested, ""
        if status in ("loading", "unloaded"):
            return None, f"Model '{requested}' is {status}. Load it first: node execute.mjs load {requested}"
        if not router:  # a server that reports no status: take the name as given
            return requested, ""
        if loaded:
            return loaded[0], f"'{requested}' is not offered here, so using the loaded model '{loaded[0]}'."
        return None, f"Model '{requested}' is not offered and none is loaded. Available: {', '.join(statuses) or 'none'}"

    if loaded:
        return loaded[0], ""
    if statuses and not router:  # no status reported (Ollama): start with the first one listed
        return next(iter(statuses)), ""
    return None, "No model is loaded. Load one first: node execute.mjs load <model>  (see: python -m tiny_agent.llm_client.check)"


def chat(llm: LLM, system: str | None = None, input_fn=None, output=print) -> None:
    """Run the interactive chat until the user leaves.

    The whole conversation is sent on every turn, so the model remembers it. If a
    turn fails or is cancelled, the unanswered question is dropped from the history
    (two ``user`` messages in a row are rejected by many chat templates).

    Args:
        llm: The model to talk to. ``/model`` and ``/think`` change it in place.
        system: Optional system message that sets the model's role.
        input_fn: Reads one line; ``None`` (the default) uses the built-in ``input``,
            looked up when the chat starts. Replaceable so tests can script a conversation.
        output: Prints one line; replaceable for the same reason.
    """
    input_fn = input_fn or input
    # The conversation memory: resent in full on every turn because the server is stateless.
    # A system message, when given, stays first so /clear can keep it.
    messages = [{"role": "system", "content": system}] if system else []
    output(f"TinyAgent chat. Model: {llm.model} at {llm.base_url}. Type /help for commands, /exit to quit.")
    log.info("chat started: model=%s url=%s think=%s system=%r", llm.model, llm.base_url, llm.think, system)

    while True:
        try:
            line = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output("")
            log.info("chat ended (Ctrl-D or Ctrl-C)")
            return
        if not line:
            continue

        if line.startswith("/"):
            command, _, argument = line[1:].partition(" ")
            argument = argument.strip()
            log.info("command: /%s %s", command, argument)
            if command in ("exit", "quit"):
                log.info("chat ended (/%s)", command)
                return
            if command == "help":
                output(HELP)
            elif command == "clear":
                messages = messages[:1] if system else []
                output("Conversation cleared.")
            elif command == "models":
                statuses = check.model_statuses(llm.base_url)
                if statuses is None:
                    output(f"Cannot reach the model server at {llm.base_url}.")
                for name, status in (statuses or {}).items():
                    output(f"{'*' if name == llm.model else ' '} {name:<20} {status}")
            elif command == "model":
                if not argument:
                    output(f"Current model: {llm.model}")
                else:
                    model, note = pick_model(argument, llm.base_url)
                    if model is None:
                        output(note)
                    else:
                        llm.model = model
                        output(note or f"Now using {model}.")
            elif command == "think":
                if argument not in ("on", "off"):
                    output(f"Thinking is {'on' if llm.think else 'off'}. Use /think on or /think off.")
                else:
                    llm.think = argument == "on"
                    output(f"Thinking is {argument}.")
            else:
                output(f"Unknown command: /{command}. Type /help.")
            continue

        messages.append({"role": "user", "content": line})
        log.info("user: %s", line)
        try:
            response = llm.generate(messages)
        except RuntimeError as error:
            messages.pop()
            output(f"error: {error}")
            log.error("model call failed: %s", error)
            continue
        except KeyboardInterrupt:
            messages.pop()
            output("\n(cancelled)")
            log.warning("turn cancelled by the user")
            continue

        text = response.content or ""
        if not text:
            messages.pop()
            output("(the model sent no text)")
            log.warning("the model sent no text (tool_call=%s)", response.tool_call)
            continue
        if response.reasoning and llm.think:
            output(f"[thinking] {response.reasoning.strip()}")
            log.debug("reasoning: %s", response.reasoning)
        output(f"{llm.model}> {text.strip()}")
        metadata = response.metadata or {}
        log.info("assistant (%s, %s prompt + %s completion tokens): %s", metadata.get("model", llm.model),
                 metadata.get("prompt_tokens"), metadata.get("completion_tokens"), text.strip())
        messages.append({"role": "assistant", "content": text})


def truthy(value: str | None) -> bool:
    """True for the usual ways to say yes in a settings file: 1, true, yes or on."""
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def build_parser() -> argparse.ArgumentParser:
    """The command-line options."""
    parser = argparse.ArgumentParser(prog="tiny-agent", description="Chat with the TinyAgent model from the terminal.")
    parser.add_argument("message", nargs="*", help="ask one question, print the answer and exit (no interactive chat)")
    parser.add_argument("-m", "--model", default=os.environ.get("TINYAGENT_MODEL"),
                        help="model name (default: the model that is loaded; or $TINYAGENT_MODEL)")
    parser.add_argument("--url", default=os.environ.get("TINYAGENT_LLM_URL", check.DEFAULT_URL),
                        help=f"API root ending in /v1 (default: {check.DEFAULT_URL}; or $TINYAGENT_LLM_URL)")
    parser.add_argument("--think", action="store_true", default=truthy(os.environ.get("TINYAGENT_THINK")),
                        help="let the model reason before answering (slower; or TINYAGENT_THINK=1)")
    parser.add_argument("--system", default=os.environ.get("TINYAGENT_SYSTEM"),
                        help="a system message that sets the model's role (or $TINYAGENT_SYSTEM)")
    parser.add_argument("--demo", action="store_true", help="run main.py's demo first, then chat")
    return parser


def main(argv: list[str] | None = None) -> int:
    """The ``tiny-agent`` command. Returns the process exit code.

    Args:
        argv: Arguments after the program name; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` on success; ``1`` if a one-shot question failed; ``2`` if no model
        can be used (server down, nothing loaded, unknown model) or the options
        were wrong.
    """
    loaded = env.load_env()  # .env fills in settings the real environment does not set
    args = build_parser().parse_args(argv)
    logs.setup("cli")
    log.info("tiny-agent started: argv=%s", sys.argv[1:] if argv is None else argv)
    if loaded:
        log.info("settings loaded from the .env file: %s", ", ".join(sorted(loaded)))  # names only, never values
    # Model replies often contain emoji; Windows consoles default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue] (the hasattr check above guards it)
    try:  # arrow keys and history in the prompt, where the platform has readline
        import readline  # noqa: F401
    except ImportError:
        pass

    if args.demo:
        from tiny_agent import main as demo  # imported here: only needed for --demo

        log.info("running the main.py demo first")
        try:
            demo.main()
        except RuntimeError as error:
            print(f"demo failed: {error}")
            log.error("demo failed: %s", error)
        print()

    model, note = pick_model(args.model, args.url)
    if model is None:
        print(note)
        log.error("no usable model: %s", note)
        return 2
    if note:
        print(note)
        log.warning(note)
    log.info("using model %s at %s (think=%s)", model, args.url, args.think)
    try:
        api_key = env.api_key_from_env()  # moved to tiny_agent.llm_client.env so main.py need not import this file
    except (OSError, ValueError) as error:
        print(f"Cannot read the API key file: {error}")
        log.error("cannot read the API key file: %s", error)
        return 2
    # LLM keeps its own placeholder key unless one is configured.
    llm = LLM(model=model, base_url=args.url, think=args.think, **({"api_key": api_key} if api_key else {}))

    if args.message:
        question = " ".join(args.message)
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
            {"role": "user", "content": question}
        ]
        log.info("one-shot question: %s", question)
        try:
            response = llm.generate(messages)
        except RuntimeError as error:
            print(f"error: {error}")
            log.error("model call failed: %s", error)
            return 1
        print(response.content or "(the model sent no text)")
        metadata = response.metadata or {}
        log.info("one-shot answer (%s, %s prompt + %s completion tokens): %s", metadata.get("model", llm.model),
                 metadata.get("prompt_tokens"), metadata.get("completion_tokens"), response.content)
        return 0

    chat(llm, system=args.system)
    return 0


if __name__ == "__main__":
    sys.exit(main())
