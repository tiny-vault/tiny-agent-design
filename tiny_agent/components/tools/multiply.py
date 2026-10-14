def multiply(a: str, b: str) -> float:
    """Multiply two numbers given as text, because a model writes its tool arguments as strings.

    Arguments:
        a: The first number.
        b: The second number.
    """
    return float(a) * float(b)
