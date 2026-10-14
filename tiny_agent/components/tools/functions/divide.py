"""Division as a callable tool: text arguments become a quotient.

A model requests an action; the harness runs the registered Python function.
``divide`` converts its dividend ``a`` and divisor ``b`` from text to floats
and returns ``a / b``. For example, ``divide("42", "6")`` returns ``7.0``.
Annotated ``float | str``, like ``multiply``: the learned schema says ``number``, text is converted.

This module only implements arithmetic. It makes no model calls and does not
register itself. The complete happy-path demo registers it for the learned
calling stage after multiplication. The number can then be wrapped in an
observation and shown to the model in a later run.

A zero divisor raises ``ZeroDivisionError`` and non-numeric text raises
``ValueError``; those failures are not converted into observations here.
Results follow floating-point arithmetic rather than exact decimal division.
Compare ``multiply.py`` for the same conversion pattern without a divisor.
"""

def divide(a: float | str, b: float | str) -> float:
    """Divide two numbers. They may also arrive as text, which is converted, so the learned schema says ``number``.

    Args:
        a: The dividend, the number to be divided, such as 42 (or numeric text such as "42").
        b: The divisor, the number to divide by, such as 6; it must not be zero.

    Returns:
        The floating-point quotient.

    Raises:
        ValueError: Either text cannot be converted to a float.
        ZeroDivisionError: ``b`` is zero.
    """
    return float(a) / float(b)
