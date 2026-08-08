import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        for i in range(n):
            xi = X[i]
            z = np.dot(w, xi) + b
            y_hat = 1.0 if z >= 0 else 0.0
            error = y[i] - y_hat
            w += lr * error * xi
            b += lr * error

    return (w.tolist(), float(b))