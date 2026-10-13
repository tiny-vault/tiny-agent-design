"""Run scripts from ``experimentation/illustrated_agents`` by name, or pick them in a popup.

Usage, from the repo root::

    python experimentation/trigger_ias.py                          popup REPL: pick a script, it runs, the popup comes back
    python experimentation/trigger_ias.py --list                   print the scripts
    python experimentation/trigger_ias.py shot_prompting/zero_shot run one and exit (the .py suffix is optional)

In the popup, type to filter, press Enter or double-click to run the selected script, and press Esc or Cancel to quit. Output goes to the terminal that started this program. If a popup cannot be shown (no display), the same loop runs as a numbered prompt in the terminal.

Any ``.py`` file added to the folder or its subfolders is picked up automatically; only ``__init__`` is skipped. A script runs exactly as ``python -m experimentation.illustrated_agents.<name>`` would, so its ``if __name__ == "__main__":`` block executes.

Exit codes:
    * ``0``: the script ran, the list was printed, or the REPL was closed.
    * ``2``: more than one argument, or an unknown script name (the list is printed).
    * In one-shot mode, anything a script raises propagates with its own traceback and a non-zero exit. In the REPL it is printed and the loop continues.
"""

import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
"""The repo root; put on ``sys.path`` so ``experimentation.illustrated_agents`` imports from anywhere."""

PACKAGE = "experimentation.illustrated_agents"
"""Dotted package path the scripts are run under, relative to the repo root."""

FOLDER = Path(__file__).parent / "illustrated_agents"
"""Folder scanned for scripts.

:meta hide-value:
"""

HELPERS = {"__init__"}
"""Module names in :data:`FOLDER` that are not runnable scripts."""


def scripts() -> list[str]:
    """List the runnable script names.

    Returns:
        The ``.py`` files in :data:`FOLDER` and its subfolders as names without the suffix, minus :data:`HELPERS`, sorted alphabetically. A script in a subfolder is named ``folder/script`` (for example ``["agent_without_memory", "best_of_n/orm", ...]``).
    """
    found = (p.relative_to(FOLDER).with_suffix("").as_posix() for p in FOLDER.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(name for name in found if name not in HELPERS)


def run(name: str) -> None:
    """Run one script as ``__main__``.

    Args:
        name: A name from :func:`scripts`, such as ``best_of_n/orm``.
    """
    # The repo root must be importable for the dotted module path; a script run by path would not have it
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    runpy.run_module(f"{PACKAGE}.{name.replace('/', '.')}", run_name="__main__")


def popup_choice(names: list[str]) -> str | None:
    """Show a popup listing the scripts and wait for a choice.

    Args:
        names: The script names to offer.

    Returns:
        The chosen name, or ``None`` if the popup was cancelled.

    Raises:
        Exception: Whatever :mod:`tkinter` raises when no display is available; the caller falls back to the terminal.
    """
    import tkinter as tk

    root = tk.Tk()
    root.title("Run an illustrated_agents script")
    root.geometry("460x420")
    # Keep the popup in front of the terminal so it is not lost behind other windows
    root.attributes("-topmost", True)
    chosen: list[str] = []

    tk.Label(root, text="Type to filter, Enter or double-click to run, Esc to quit").pack(padx=10, pady=(10, 4), anchor="w")
    query = tk.StringVar()
    entry = tk.Entry(root, textvariable=query)
    entry.pack(fill="x", padx=10)
    box = tk.Listbox(root, activestyle="dotbox", exportselection=False)
    box.pack(fill="both", expand=True, padx=10, pady=8)

    def refill(*_: object) -> None:
        # Show only names containing the typed text, with the first one selected
        box.delete(0, "end")
        for name in names:
            if query.get().lower() in name.lower():
                box.insert("end", name)
        if box.size():
            box.selection_set(0)

    def accept(*_: object) -> None:
        if box.curselection():
            chosen.append(box.get(box.curselection()[0]))
            root.destroy()

    def move(step: int) -> str:
        # Up and Down keep keyboard focus in the entry while moving the selection
        if box.size():
            current = box.curselection()[0] if box.curselection() else 0
            new = max(0, min(box.size() - 1, current + step))
            box.selection_clear(0, "end")
            box.selection_set(new)
            box.see(new)
        return "break"

    buttons = tk.Frame(root)
    buttons.pack(pady=(0, 10))
    tk.Button(buttons, text="Run", width=10, command=accept).pack(side="left", padx=4)
    tk.Button(buttons, text="Cancel", width=10, command=root.destroy).pack(side="left", padx=4)

    query.trace_add("write", refill)
    entry.bind("<Return>", accept)
    entry.bind("<Down>", lambda _: move(1))
    entry.bind("<Up>", lambda _: move(-1))
    box.bind("<Double-Button-1>", accept)
    root.bind("<Escape>", lambda _: root.destroy())
    refill()
    entry.focus_force()
    root.mainloop()
    return chosen[0] if chosen else None


def terminal_choice(names: list[str]) -> str | None:
    """Offer the scripts as a numbered list in the terminal (the fallback when no popup can be shown).

    Args:
        names: The script names to offer.

    Returns:
        The chosen name, or ``None`` on an empty answer, ``q`` or end of input.
    """
    for i, name in enumerate(names, 1):
        print(f"{i:>3}. {name}")
    while True:
        try:
            answer = input("Number or name (empty to quit): ").strip().removesuffix(".py")
        except EOFError:
            return None
        if answer in ("", "q"):
            return None
        if answer in names:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print("Not in the list.")


def repl(names: list[str]) -> int:
    """Ask which script to run, run it, and ask again until the popup is cancelled.

    Args:
        names: The script names to offer.

    Returns:
        ``0``.
    """
    choose = popup_choice
    while True:
        try:
            name = choose(names)
        except Exception as error:
            # No display (or no tkinter): keep the same loop in the terminal
            print(f"Popup unavailable ({error}); using the terminal.")
            choose = terminal_choice
            continue
        if name is None:
            return 0
        print(f"\n===== {name} =====", flush=True)
        try:
            run(name)
        except (Exception, SystemExit):
            # A failing script must not end the session: show why, then ask again
            traceback.print_exc()
        print(flush=True)


def main(argv: list[str]) -> int:
    """Run the script named in ``argv``, list the scripts, or start the popup REPL.

    Args:
        argv: Command-line arguments without the program name. Empty starts the REPL; ``--list`` prints the scripts; one other item is a script name, with or without ``.py``.

    Returns:
        The process exit code, ``0`` on success, listing or REPL exit, ``2`` for bad usage or an unknown name.
    """
    names = scripts()
    if not argv:
        return repl(names)
    if argv == ["--list"]:
        print("\n".join(names))
        return 0
    if len(argv) != 1:
        print("Usage: python experimentation/trigger_ias.py [--list | <script>]\n\nScripts:\n  " + "\n  ".join(names))
        return 2
    name = argv[0].removesuffix(".py")
    if name not in names:
        print(f"Unknown script '{name}'. Scripts:\n  " + "\n  ".join(names))
        return 2
    run(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
