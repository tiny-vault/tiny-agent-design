"""Send one chat message to a local Ollama model, using only the stdlib.

Uses Ollama's OpenAI-compatible endpoint. The request shape follows the
example in *An Illustrated Guide to AI Agents* by Maarten Grootendorst and
Jay Alammar (O'Reilly).

Usage:
    python examples/chat_ollama.py [model] [message]

Examples:
    python examples/chat_ollama.py gemma3:4b
    python examples/chat_ollama.py gemma4:e4b-it-qat "Hi! How's life?"
"""

import json
import sys
import urllib.error
import urllib.request

URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:e4b-it-qat"
DEFAULT_MESSAGE = "Hi! How's life?"


def chat(model: str, message: str) -> dict:
    """Send one user message and return the first choice of the reply.

    Args:
        model: Ollama model tag, for example ``gemma3:4b``.
        message: The user message.

    Returns:
        The first element of the response's ``choices`` list.

    Raises:
        urllib.error.URLError: If the Ollama server cannot be reached.
        urllib.error.HTTPError: If the server rejects the request, for
            example because the model is not pulled.
    """
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": message}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        url=URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    return result["choices"][0]


def main() -> None:
    """Read the model and message from the command line and print the reply."""
    sys.stdout.reconfigure(encoding="utf-8")  # replies contain emoji; Windows consoles default to cp1252
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    try:
        choice = chat(model, message)
    except urllib.error.HTTPError as error:
        sys.exit(f"Ollama rejected the request ({error.code}). Is '{model}' pulled?")
    except urllib.error.URLError as error:
        sys.exit(f"Cannot reach Ollama at {URL}: {error.reason}")
    print(choice)


if __name__ == "__main__":
    main()
