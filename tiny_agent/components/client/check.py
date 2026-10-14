"""Find out which models the server has loaded.

The Docker gateway runs llama.cpp in router mode: it offers several models but
keeps only one loaded at a time, and refuses to chat with a model that is not
loaded. These helpers ask the server so the agent can fail early with a clear
message instead of an HTTP error.

Why it exists: a local GPU can hold only one big model at a time. If you ask the
router for a model that is not loaded you get an error that does not say what to
do; asking first lets us answer "load it with ``node execute.mjs load <model>``".

How it works: ``GET <base_url>/models`` returns a list of models. A router adds a
``status.value`` to each (``loaded``, ``loading`` or ``unloaded``). Servers that
do not report a status (Ollama) get ``"unknown"``, and are never blocked. An
unreachable server gives ``None``, which is different from "nothing is loaded".

Functions: :func:`model_statuses` is the one request; :func:`loaded_models` and
:func:`model_status` are small views of its answer; :func:`main` is the command line.

Run it from the repo root to see the state::

    python -m tiny_agent.components.client.check                 # list every model and its status
    python -m tiny_agent.components.client.check gemma4-e4b      # exit code 0 if loaded, 1 if not

Libraries: click and httpx.
"""

import sys

import click
import httpx

from tiny_agent.config.settings import DEFAULT_LLM_URL, Settings

DEFAULT_URL = DEFAULT_LLM_URL  # the Docker gateway (see tiny_agent/config/settings.py)


def model_statuses(base_url: str = DEFAULT_URL, timeout: float = 5) -> dict[str, str] | None:
    """Return ``{model: status}`` for every model the server offers.

    Status is ``"loaded"``, ``"loading"`` or ``"unloaded"`` on a router. A server
    that does not report status (Ollama, a single-model llama.cpp) gives
    ``"unknown"`` for each model.

    Args:
        base_url: The API root, ending in ``/v1``.
        timeout: Seconds to wait for the server.

    Returns:
        The statuses, or ``None`` if the server cannot be reached or answers
        with something that is not a model list.
    """
    try:
        reply = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        reply.raise_for_status()
        data = reply.json().get("data", [])
        return {m["id"]: (m.get("status") or {}).get("value", "unknown") for m in data}
    # HTTPError covers refused/reset connections, timeouts and HTTP error statuses; InvalidURL and ValueError
    # cover a malformed URL and a reply that is not JSON; the rest cover JSON that is not shaped like a model list.
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError, KeyError, AttributeError):
        return None


def loaded_models(base_url: str = DEFAULT_URL) -> list[str]:
    """Return the models that are loaded right now (empty if none or unreachable).

    Only a router reports ``"loaded"``; a server with no status (Ollama) always
    gives an empty list here, because it cannot say.

    Args:
        base_url: The API root, ending in ``/v1``.

    Returns:
        The names of the loaded models, in the server's order. An empty list
        means either nothing is loaded or the server could not be reached; use
        :func:`model_statuses` to tell those apart.
    """
    return [m for m, status in (model_statuses(base_url) or {}).items() if status == "loaded"]


def model_status(model: str, base_url: str = DEFAULT_URL) -> str | None:
    """Return one model's status, or ``None`` if the server is unreachable or does not offer it.

    This is what :meth:`tiny_agent.components.client.llm.LLM.generate` calls before sending a
    chat request.

    Args:
        model: The model name to look up, as the server lists it.
        base_url: The API root, ending in ``/v1``.

    Returns:
        ``"loaded"``, ``"loading"``, ``"unloaded"`` or ``"unknown"`` (the server
        does not report status), or ``None`` when the server cannot be reached or
        does not list ``model``.
    """
    return (model_statuses(base_url) or {}).get(model)


# The default is a function, so the settings (environment, .env file) are read when the command runs, not at import.
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("model", required=False)
@click.option("--url", default=lambda: Settings.from_env().llm_url, help="the API root to query, ending in /v1 (default: $TINYAGENT_LLM_URL or the Docker gateway)")
def command(model: str | None, url: str) -> int:
    """Print the model list, or check one MODEL.

    With no MODEL it prints every model with its status, marking the loaded one with *. With a MODEL it prints that
    model's status only, so it can be used in scripts.
    """
    base_url = url
    statuses = model_statuses(base_url)
    if statuses is None:
        print(f"Cannot reach the model server at {base_url}. Is the stack running? (node execute.mjs up)")
        return 2

    if model:
        status = statuses.get(model)
        if status is None:
            print(f"'{model}' is not offered. Available: {', '.join(statuses) or 'none'}")
            return 1
        print(f"{model}: {status}")
        return 0 if status in ("loaded", "unknown") else 1

    for model, status in statuses.items():
        print(f"{'*' if status == 'loaded' else ' '} {model:<20} {status}")
    loaded = [m for m, s in statuses.items() if s == "loaded"]
    print(f"\nLoaded: {', '.join(loaded) if loaded else 'none (load one: node execute.mjs load <model>)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command and return its process exit code.

    Command line::

        python -m tiny_agent.components.client.check [MODEL] [--url BASE_URL]

    Args:
        argv: The arguments after the program name. ``None`` (the default) reads them from ``sys.argv``.

    Returns:
        The process exit code, one of the following values.

        * ``0``: the list was printed, or the named model is loaded (or its
          status is unknown because the server does not report it).
        * ``1``: the named model is not offered, or is ``unloaded`` or ``loading``.
        * ``2``: the server could not be reached, or the command was used wrongly
          (``--url`` without a value, an unknown option, more than one model name); the usage message is on stderr.

        ``-h`` or ``--help`` prints the help and returns ``0``.
    """
    try:
        code = command.main(args=argv, prog_name="python -m tiny_agent.components.client.check", standalone_mode=False)
    except click.ClickException as error:  # wrong options
        error.show()
        return error.exit_code
    return code if isinstance(code, int) else 0  # --help ends the command with code 0 and no return value


if __name__ == "__main__":
    sys.exit(main())
