"""Subtract as a callable tool: text arguments become a difference.

A model requests an action; the harness runs the registered Python function.
``subtract`` converts ``a`` and ``b`` from text to floats and returns ``a - b``. For example,
``subtract("45", "3")`` returns ``42.0``.
Annotated ``float | str``, like ``multiply``: the learned schema says ``number``, text is converted.

This module only implements arithmetic. It makes no model calls and does not
register itself; the ReAct tests register it next to ``multiply``. Non-numeric
text raises ``ValueError``; that failure is not converted into an observation here.
Results follow floating-point arithmetic, so decimals such as 4.6 may show rounding noise.
Compare ``multiply.py`` for the same conversion pattern.
"""


def subtract(a: float | str, b: float | str) -> float:
    """Subtract two numbers. They may also arrive as text, which is converted, so the learned schema says ``number``.

    Args:
        a: The number to subtract from, such as 45 (or numeric text such as "45").
        b: The number to take away, such as 3 (or numeric text such as "3").

    Returns:
        The floating-point difference.

    Raises:
        ValueError: Either text cannot be converted to a float.
    """
    return float(a) - float(b)
