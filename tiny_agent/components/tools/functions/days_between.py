"""Days between as a callable tool: exact calendar arithmetic, which models often get wrong by a day or two.

Counting days across months of different lengths and leap years is easy for code and error-prone for a model.
``days_between("2026-10-15", "2026-12-25")`` returns ``71``: the days from ``a`` to ``b``, negative when ``b`` is
earlier. Both dates are ISO text (``YYYY-MM-DD``), the format ``today`` returns.

Text that is not an ISO date raises ``ValueError``; in the ReAct loop that becomes an observation the model can
correct. No model calls. Compare ``today.py``.
"""

import datetime


def days_between(a: str, b: str) -> int:
    """Days between two ISO dates.

    Args:
        a: The start date, such as 2026-10-15.
        b: The end date, such as 2026-12-25.
    """
    return (datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days
