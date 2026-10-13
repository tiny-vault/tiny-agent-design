"""Load settings from a ``.env`` file into the process environment.

A ``.env`` file is a plain text list of ``NAME=value`` lines. By default this module
reads ``code/.env`` (the file next to the ``llm_client`` and ``tiny_agent`` folders,
which is git-ignored), so settings such as the model name or the server address do
not have to be typed on every command.

Why it exists: the model name, server address and API key change per machine and
must not be hard-coded or committed. Environment variables are the usual place for
such settings; a ``.env`` file is a convenient way to set them without typing them
each time. Real code would use the ``python-dotenv`` package; this is a few lines of
standard library instead, so the project keeps zero dependencies.

How it works: ``load_env`` reads the file, ``parse`` turns its text into a
dict (using ``_parse_value`` for quotes and comments), and each name not already
in ``os.environ`` is copied in. Other code then reads it with ``os.environ.get``.

Rules, kept deliberately small:

* A value that is already set in the real environment always wins. The file only
  fills in what is missing, so ``TINYAGENT_MODEL=x tiny-agent`` still overrides it.
* A missing, empty or unreadable file is not an error: nothing is loaded.
* Lines that cannot be understood are skipped, never fatal.
* Nothing from the file is ever printed or logged by this module.

Supported syntax::

    # a comment
    TINYAGENT_MODEL=gemma4-e4b
    export TINYAGENT_LLM_URL=http://localhost:3200/v1   # a trailing comment
    TINYAGENT_SYSTEM="Be brief.\\nAnswer in one line."
    TINYAGENT_API_KEY='single quotes keep everything literally'

Set ``TINYAGENT_ENV_FILE`` in the real environment to read a different file.

Standard library only.
"""

import os
import re
from pathlib import Path

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def default_path() -> Path:
    """Return the file to read: ``$TINYAGENT_ENV_FILE`` if set, else ``code/.env``."""
    configured = os.environ.get("TINYAGENT_ENV_FILE")
    return Path(configured) if configured else Path(__file__).resolve().parents[1] / ".env"


def parse(text: str) -> dict[str, str]:
    """Turn the text of a ``.env`` file into a ``{name: value}`` dict.

    Args:
        text: The whole file.

    Returns:
        The settings, in file order. When a name appears twice the last one wins.
        Blank lines, comments and lines without a valid ``NAME=`` are skipped.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, separator, rest = line.partition("=")
        name = name.strip()
        if not separator or not _KEY.fullmatch(name):
            continue
        values[name] = _parse_value(rest.strip())
    return values


def _parse_value(rest: str) -> str:
    """Return the value from the text after ``=``: unquote it, or cut a trailing comment."""
    if rest[:1] == '"':
        out, i = [], 1
        while i < len(rest) and rest[i] != '"':
            if rest[i] == "\\" and i + 1 < len(rest):
                out.append(_ESCAPES.get(rest[i + 1], "\\" + rest[i + 1]))
                i += 2
            else:
                out.append(rest[i])
                i += 1
        return "".join(out)
    if rest[:1] == "'":
        end = rest.find("'", 1)
        return rest[1:end] if end != -1 else rest[1:]
    match = re.search(r"\s#", rest)  # a comment needs a space before the #, so a#b is kept whole
    return (rest[: match.start()] if match else rest).strip()


def load_env(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Read a ``.env`` file and put its settings into ``os.environ``.

    Args:
        path: The file to read. ``None`` means :func:`default_path`.
        override: ``False`` (the default) leaves names that are already set in the
            real environment alone. ``True`` lets the file replace them.

    Returns:
        The settings that were actually set (not the ones skipped because the real
        environment already had them). Empty if the file is missing or unreadable.
    """
    target = Path(path) if path is not None else default_path()
    try:
        text = target.read_text(encoding="utf-8-sig")  # tolerates a byte-order mark
    except (OSError, UnicodeDecodeError):
        return {}
    loaded = {}
    for name, value in parse(text).items():
        if override or name not in os.environ:
            os.environ[name] = value
            loaded[name] = value
    return loaded
