import numpy as np

def local_vjp(operation, inputs, output, upstream_gradient):
    """
    Returns: one gradient contribution per input in input order.
    """
    g = np.float64(upstream_gradient)
    t = np.float64(output)

    if operation == "add":
        return [float(g), float(g)]

    elif operation == "mul":
        a, b = np.float64(inputs[0]), np.float64(inputs[1])
        return [float(g * b), float(g * a)]

    elif operation == "tanh":
        return [float(g * (1 - t * t))]