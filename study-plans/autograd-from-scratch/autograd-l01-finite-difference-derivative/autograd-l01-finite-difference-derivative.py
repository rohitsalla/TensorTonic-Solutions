import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    coeffs = np.array(coefficients, dtype=np.float64)
    x  = np.float64(x)
    h  = np.float64(h)

    # Evaluate polynomial using Horner's method for numerical stability
    # coefficients are in ascending power order: c0 + c1*x + c2*x^2 + ...
    # np.polyval expects DESCENDING order, so reverse first
    fx   = np.polyval(coeffs[::-1], x)
    fxh  = np.polyval(coeffs[::-1], x + h)

    slope = (fxh - fx) / h

    return (float(fx), float(fxh), float(slope))