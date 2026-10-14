"""Multiplication as a callable tool: text arguments become a numeric result.

A tool gives the harness a deterministic operation outside model text
generation. ``multiply`` is annotated ``float | str``: the learned-tools schema
advertises ``number`` parameters, and the prompting variant's model writes its
arguments as text, so it converts each with ``float`` and returns their product. For example, ``multiply("6", "7")`` returns ``42.0``.

This file does not talk to a model or register itself. ``default_tools`` and
``default_native_tools`` attach it to their registries, whose descriptions or
schemas explain how a model can request it. The harness executes it after
parsing an action.

Whitespace and scientific notation follow Python's float conversion.
Non-numeric text raises ``ValueError``; rounding and non-finite values follow
floating-point arithmetic rather than exact decimal calculation. Compare
``divide.py`` for a tool with an additional division-by-zero failure.
"""

def multiply(a: float | str, b: float | str) -> float:
    """Multiply two numbers. They may also arrive as text, which is converted, so the learned schema says ``number``.

    Args:
        a: The first factor, a number such as 6 (or numeric text such as "6").
        b: The second factor, a number such as 7 (or numeric text such as "7").

    Returns:
        The floating-point product.

    Raises:
        ValueError: Either text cannot be converted to a float.
    """
    return float(a) * float(b)
