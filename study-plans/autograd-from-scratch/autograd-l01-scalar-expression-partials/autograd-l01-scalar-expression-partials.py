import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    a, b, c, h = np.float64(a), np.float64(b), np.float64(c), np.float64(h)

    def f(a, b, c):
        return a * b + c

    d = f(a, b, c)

    partial_a = (f(a + h, b, c) - d) / h
    partial_b = (f(a, b + h, c) - d) / h
    partial_c = (f(a, b, c + h) - d) / h

    return (float(d), float(partial_a), float(partial_b), float(partial_c))