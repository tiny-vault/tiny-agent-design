"""Log files for the two terminal tools, ``cli.py`` and ``main.py``.

Each tool writes its own file under ``experimentation/chat-app/logs/`` (``cli.log`` and
``main.log``). Nothing else logs here: the backend, llama.cpp and nginx
already write their logs in Docker (``node execute.mjs logs <service>``); OpenObserve collects those and these files in one place, and the library modules in ``tiny_agent`` stay silent;
the tools log what those modules return or raise.

Nothing is printed to the terminal by this module (the tools do their own
printing), and a problem with logging, such as a read-only disk, never stops the
tool: it just runs without a log.

Settings (environment variables):

* ``TINYAGENT_LOG_DIR``: the folder to write to instead of ``<repo>/experimentation/chat-app/logs``.
* ``TINYAGENT_LOG_LEVEL``: ``DEBUG``, ``INFO`` (default), ``WARNING`` or ``ERROR``.

Files rotate at 1 MB and keep 3 old copies (``cli.log.1`` and so on).

Why files, and why this way: a terminal chat cannot also show log lines, so the
tools write them to a file instead. Each tool gets its own named logger
(``tiny_agent.cli``, ``tiny_agent.main``) that does not pass records to the root
logger, which is why nothing leaks to the screen. Records are only written when
a tool calls :func:`setup`; library code never configures logging. Logs name the
settings used, never secret values.

Example:
    >>> log = setup("cli")        # doctest: +SKIP
    >>> log.info("started")       # doctest: +SKIP
"""

import logging
import logging.handlers
from pathlib import Path

from tiny_agent.config.settings import Settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MARK = "_tiny_agent_file_handler"  # tags the handlers this module adds, so setup() can replace them


def log_dir() -> Path:
    """Return the folder the log files go in.

    ``$TINYAGENT_LOG_DIR`` if set. Otherwise ``experimentation/chat-app/logs/`` in the repo
    when the code is running from a checkout (the folder that holds
    ``pyproject.toml``), and ``logs/`` in the current directory when it is not (a
    regular, non-editable install).
    """
    configured = Settings.from_env().log_dir
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[2]  # tiny_agent/logging/log.py -> the repo root
    if (root / "pyproject.toml").exists():
        return root / "experimentation" / "chat-app" / "logs"
    return Path.cwd() / "logs"


def setup(name: str) -> logging.Logger:
    """Start logging to ``<log_dir>/<name>.log`` and return the logger.

    Safe to call more than once: a second call replaces the file handler instead of
    adding another, so lines are never written twice. The logger does not pass
    records on to the root logger, so nothing reaches the terminal.

    Args:
        name: ``"cli"`` or ``"main"``. It names the file and the logger
            (``tiny_agent.<name>``).

    Returns:
        The logger. If the log file cannot be opened, it is returned with no file
        handler and everything sent to it is quietly dropped.
    """
    logger = logging.getLogger(f"tiny_agent.{name}")
    logger.propagate = False
    logger.setLevel(getattr(logging, Settings.from_env().log_level.upper(), logging.INFO))

    for handler in [h for h in logger.handlers if getattr(h, _MARK, False)]:
        logger.removeHandler(handler)
        handler.close()

    try:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            directory / f"{name}.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError:
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    handler.setFormatter(logging.Formatter(FORMAT))
    setattr(handler, _MARK, True)
    logger.addHandler(handler)
    return logger
