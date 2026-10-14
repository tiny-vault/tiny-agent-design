"""Today as a callable tool: the local calendar date, which a model cannot know on its own.

A model's knowledge stops at its training data, so "what is the date?" is a fact it can only guess. ``today`` asks
the computer's clock and returns the date as ISO text, such as ``"2026-10-15"``, which ``days_between`` reads back.

The date is the machine's local date (in Docker, the container's time zone, often UTC), so near midnight it can
differ from the user's. No arguments, no model calls. Compare ``days_between.py``, the other date tool.
"""

import datetime


def today() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()
