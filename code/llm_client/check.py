"""Find out which models the server has loaded.

The Docker gateway runs llama.cpp in router mode: it offers several models but
keeps only one loaded at a time, and refuses to chat with a model that is not
loaded. These helpers ask the server so the agent can fail early with a clear
message instead of an HTTP error.

Run it from the repo root to see the state::

    python -m llm_client.check                 # list every model and its status
    python -m llm_client.check gemma4-e4b      # exit code 0 if loaded, 1 if not

Standard library only.
"""

import http.client
import json
import os
import sys
import urllib.request

from llm_client import env

DEFAULT_URL = "http://localhost:3200/v1"  # the Docker gateway


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
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout) as response:
            data = json.loads(response.read()).get("data", [])
        return {m["id"]: (m.get("status") or {}).get("value", "unknown") for m in data}
    # OSError covers refused/reset connections, timeouts and HTTP errors; ValueError
    # covers a malformed URL and a reply that is not JSON; the rest cover JSON that
    # is not shaped like a model list.
    except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError, AttributeError):
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

    This is what :meth:`llm_client.llm.LLM.generate` calls before sending a
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


def main(argv: list[str] | None = None) -> int:
    """Print the model list, or check one model. Returns the process exit code.

    Command line::

        python -m llm_client.check [MODEL] [--url BASE_URL]

    With no ``MODEL`` it prints every model with its status, marking the loaded
    one with ``*``. With a ``MODEL`` it prints that model's status only, so it
    can be used in scripts.

    Args:
        argv: The arguments after the program name. ``None`` (the default) reads
            them from ``sys.argv``. ``--url`` takes the API root to query instead
            of the Docker gateway and must be followed by a value.

    Returns:
        The process exit code, one of the following values.

        * ``0``: the list was printed, or the named model is loaded (or its
          status is unknown because the server does not report it).
        * ``1``: the named model is not offered, or is ``unloaded`` or ``loading``.
        * ``2``: the server could not be reached, or the command was used wrongly
          (``--url`` without a value, an unknown option, more than one model name).

        ``-h`` or ``--help`` prints the usage line and returns ``0``.
    """
    args = list(sys.argv[1:] if argv is None else argv)  # a copy: never change the caller's list
    usage = "usage: python -m llm_client.check [MODEL] [--url BASE_URL]"
    env.load_env()  # code/.env may set TINYAGENT_LLM_URL
    base_url = os.environ.get("TINYAGENT_LLM_URL", DEFAULT_URL)
    if "-h" in args or "--help" in args:
        print(usage)
        return 0
    if "--url" in args:
        i = args.index("--url")
        if i + 1 >= len(args) or args[i + 1].startswith("-"):
            print(f"--url needs a value, for example: --url http://localhost:3200/v1\n{usage}")
            return 2
        base_url = args[i + 1]
        del args[i : i + 2]
    unknown = [a for a in args if a.startswith("-")]
    if unknown:
        print(f"Unknown option: {unknown[0]}\n{usage}")
        return 2
    if len(args) > 1:
        print(f"Give at most one model name, got {len(args)}: {', '.join(args)}\n{usage}")
        return 2

    statuses = model_statuses(base_url)
    if statuses is None:
        print(f"Cannot reach the model server at {base_url}. Is the stack running? (node execute.mjs up)")
        return 2

    if args:
        status = statuses.get(args[0])
        if status is None:
            print(f"'{args[0]}' is not offered. Available: {', '.join(statuses) or 'none'}")
            return 1
        print(f"{args[0]}: {status}")
        return 0 if status in ("loaded", "unknown") else 1

    for model, status in statuses.items():
        print(f"{'*' if status == 'loaded' else ' '} {model:<20} {status}")
    loaded = [m for m, s in statuses.items() if s == "loaded"]
    print(f"\nLoaded: {', '.join(loaded) if loaded else 'none (load one: node execute.mjs load <model>)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
