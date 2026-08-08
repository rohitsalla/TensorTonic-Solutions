import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    x = float(x)

    if activation == "relu":
        out = max(0.0, x)
        deriv = 1.0 if x > 0 else 0.0

    elif activation == "sigmoid":
        sig = 1.0 / (1.0 + np.exp(-x))
        out = sig
        deriv = sig * (1.0 - sig)

    elif activation == "tanh":
        t = np.tanh(x)
        out = t
        deriv = 1.0 - t ** 2

    elif activation == "leaky_relu":
        alpha = 0.01
        out = x if x > 0 else alpha * x
        deriv = 1.0 if x > 0 else alpha

    elif activation == "gelu":
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = 0.5 * x * (1.0 + t)
        dinner_dx = c * (1.0 + 3.0 * 0.044715 * x ** 2)
        sech2 = 1.0 - t ** 2
        deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner_dx

    elif activation == "swish":
        sig = 1.0 / (1.0 + np.exp(-x))
        out = x * sig
        deriv = sig + x * sig * (1.0 - sig)

    else:
        raise ValueError(f"Unknown activation: {activation}")

    return [round(float(out), 4), round(float(deriv), 4)]