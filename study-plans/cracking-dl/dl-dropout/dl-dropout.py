import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=float)

    if mode == "test":
        result = X
    else:
        mask = np.array(mask, dtype=float)
        result = X * mask / (1 - drop_prob)

    return np.round(result, 4).tolist()