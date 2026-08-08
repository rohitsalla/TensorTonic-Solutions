import numpy as np

def backprop(X, y, W1, b1, W2, b2):
    """
    Compute gradients for a single-hidden-layer MLP with MSE loss.
    Returns: dict with "dW1", "db1", "dW2", "db2", all rounded to 4 decimals.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)

    n = X.shape[0]

    # Forward pass
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2

    # Loss: L = (1/n) * sum((z2 - y)^2)  -> dL/dz2 = (2/n) * (z2 - y)
    dz2 = (2.0 / n) * (z2 - y)

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    relu_deriv = (z1 > 0).astype(float)
    dz1 = da1 * relu_deriv

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {
        "dW1": np.round(dW1, 4).tolist(),
        "db1": np.round(db1, 4).tolist(),
        "dW2": np.round(dW2, 4).tolist(),
        "db2": np.round(db2, 4).tolist(),
    }