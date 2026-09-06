import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    a, b, c, f, h = (np.float64(v) for v in (a, b, c, f, h))

    def forward(a, b, c, f):
        e = a * b + c
        L = e * f
        return L

    L = forward(a, b, c, f)

    # Analytic gradients via chain rule:
    # e = ab + c  ->  L = e * f
    # dL/de = f
    # dL/da = dL/de * de/da = f * b
    # dL/db = dL/de * de/db = f * a
    # dL/dc = dL/de * de/dc = f * 1 = f
    # dL/df = e = ab + c
    e = a * b + c
    analytic = [float(f * b), float(f * a), float(f), float(e)]

    # Numerical gradients via forward difference
    numerical = [
        float((forward(a + h, b, c, f) - L) / h),
        float((forward(a, b + h, c, f) - L) / h),
        float((forward(a, b, c + h, f) - L) / h),
        float((forward(a, b, c, f + h) - L) / h),
    ]

    max_gap = float(np.max(np.abs(np.array(analytic) - np.array(numerical))))

    return (float(L), analytic, numerical, max_gap)