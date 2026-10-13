"""Run one script from ``experimentation/illustrated_agents`` by name.

Usage, from the repo root::

    python trigger_ias.py            list the scripts
    python trigger_ias.py one_shot   run one (the .py suffix is optional)

Any ``.py`` file added to that folder is picked up automatically; only ``__init__`` is skipped. A script runs exactly as ``python -m experimentation.illustrated_agents.<name>`` would, so its ``if __name__ == "__main__":`` block executes.

Exit codes:
    * ``0``: the script ran, or the list was printed because no name was given.
    * ``2``: more than one argument, or an unknown script name (the list is printed).
    * Anything a script raises propagates with its own traceback and a non-zero exit.
"""

import runpy
import sys
from pathlib import Path

PACKAGE = "experimentation.illustrated_agents"
"""Dotted package path the scripts are run under, relative to the repo root."""

FOLDER = Path(__file__).parent / "experimentation" / "illustrated_agents"
"""Folder scanned for scripts.

:meta hide-value:
"""

HELPERS = {"__init__"}
"""Module names in :data:`FOLDER` that are not runnable scripts."""


def scripts() -> list[str]:
    """List the runnable script names.

    Returns:
        The stems of the ``.py`` files in :data:`FOLDER`, minus :data:`HELPERS`, sorted alphabetically (for example ``["cot_one_shot", "diff_ch1_ch2", ...]``).
    """
    return sorted(p.stem for p in FOLDER.glob("*.py") if p.stem not in HELPERS)


def main(argv: list[str]) -> int:
    """Run the script named in ``argv``, or list the scripts.

    Args:
        argv: Command-line arguments without the program name. Empty prints usage and the script list; one item is a script name, with or without ``.py``.

    Returns:
        The process exit code, ``0`` on success or when listing, ``2`` for bad usage or an unknown name.
    """
    names = scripts()
    if len(argv) != 1:
        print("Usage: python trigger_ias.py <script>\n\nScripts:\n  " + "\n  ".join(names))
        return 0 if not argv else 2
    name = argv[0].removesuffix(".py")
    if name not in names:
        print(f"Unknown script '{name}'. Scripts:\n  " + "\n  ".join(names))
        return 2
    # Scripts import their shared pieces as ``from common import ...``, which resolves when run as a file
    # (the script's folder is sys.path[0]); add that folder so it resolves here too.
    sys.path.insert(0, str(FOLDER))
    runpy.run_module(f"{PACKAGE}.{name}", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
