"""Ask the model server to load a model and wait until it is loaded.

llama.cpp in router mode keeps one model in GPU memory. This script sends ``POST /models/load`` and then polls
``GET /v1/models`` until the model's status is ``loaded``. The first load of a model downloads it from Hugging
Face (several GB, minutes); later loads take tens of seconds. It is how the llm-client folder warms up a model
before ``ask.py`` or the experimentation scripts use it (``node execute.mjs llm-client load <model>`` runs it).

The server is ``TINYAGENT_LLM_URL`` (default: this folder's own llama service). The load route is ``/models/load``
on the server root, next to ``/v1``, so the ``/v1`` suffix is cut off to find it.

Exit codes: ``0`` loaded, ``1`` the load failed or timed out, ``2`` wrong usage, no server, or unknown model.
"""

import json
import os
import sys
import time
import urllib.request

from tiny_agent.llm_client import check

USAGE = "usage: python load.py MODEL"
TIMEOUT_SECONDS = 1800  # the first load downloads the model


def main(argv: list[str]) -> int:
    """Load the model named in ``argv`` and wait for it. Returns the process exit code."""
    if len(argv) != 1 or argv[0].startswith("-"):
        print(USAGE)
        return 2
    model = argv[0]
    base_url = os.environ.get("TINYAGENT_LLM_URL", check.DEFAULT_URL).rstrip("/")
    statuses = check.model_statuses(base_url)
    if statuses is None:
        print(f"Cannot reach the model server at {base_url}. Is it running? (node execute.mjs up llm-client)")
        return 2
    if model not in statuses:
        print(f"'{model}' is not offered. Available: {', '.join(statuses) or 'none'}")
        return 2
    if statuses[model] == "loaded":
        print(f"{model} is already loaded.")
        return 0

    root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
    request = urllib.request.Request(
        f"{root}/models/load", data=json.dumps({"model": model}).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except OSError as error:
        print(f"The server refused the load: {error}")
        return 1

    print(f"Loading {model} (the first time it is downloaded from Hugging Face: minutes)...", flush=True)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = (check.model_statuses(base_url) or {}).get(model)
        if status == "loaded":
            print(f"{model} is loaded.")
            return 0
        if status == "unloaded" and time.monotonic() > deadline - TIMEOUT_SECONDS + 30:
            print(f"{model} went back to unloaded: the load failed. See: docker compose logs llama")
            return 1
        time.sleep(2)
    print(f"{model} did not finish loading in {TIMEOUT_SECONDS} s.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
