"""Add as a callable tool: text arguments become a sum.

A model requests an action; the harness runs the registered Python function.
``add`` converts ``a`` and ``b`` from text to floats and returns ``a + b``. For example,
``add("4.6", "6.5")`` returns ``11.1``.
Annotated ``float | str``, like ``multiply``: the learned schema says ``number``, text is converted.

This module only implements arithmetic. It makes no model calls and does not
register itself; the ReAct tests register it next to ``multiply``. Non-numeric
text raises ``ValueError``; that failure is not converted into an observation here.
Results follow floating-point arithmetic, so decimals such as 4.6 may show rounding noise.
Compare ``multiply.py`` for the same conversion pattern.
"""


def add(a: float | str, b: float | str) -> float:
    """Add two numbers. They may also arrive as text, which is converted, so the learned schema says ``number``.

    Args:
        a: The first number, such as 4.6 (or numeric text such as "4.6").
        b: The second number, such as 6.5 (or numeric text such as "6.5").

    Returns:
        The floating-point sum.

    Raises:
        ValueError: Either text cannot be converted to a float.
    """
    return float(a) + float(b)
