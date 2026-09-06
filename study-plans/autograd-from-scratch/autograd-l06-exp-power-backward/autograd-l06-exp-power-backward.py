import numpy as np

def exp_power_backward(x, exponent, upstream_gradient):
    """
    Returns: (exp_output, exp_gradient, power_output, power_gradient).
    """
    x  = np.float64(x)
    n  = np.float64(exponent)
    u  = np.float64(upstream_gradient)

    # Exponential: e = exp(x), gradient = u * exp(x)
    exp_output   = np.exp(x)
    exp_gradient = u * exp_output

    # Power: p = x^n, gradient = u * n * x^(n-1)
    if n == 0:
        power_output   = np.float64(1.0)
        power_gradient = np.float64(0.0)
    else:
        # For integer-valued exponents, use integer power to handle negative bases
        if n == np.floor(n):
            power_output = np.float64(x ** int(n))
        else:
            power_output = np.float64(x ** n)
        power_gradient = u * n * (x ** (n - 1))

    return (float(exp_output), float(exp_gradient),
            float(power_output), float(power_gradient))