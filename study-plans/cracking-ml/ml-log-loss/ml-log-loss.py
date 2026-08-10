import numpy as np

def log_loss(y_true, y_pred):
    """
    Returns the binary cross-entropy loss rounded to 4 decimals.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    losses = -(
        y_true * np.log(y_pred)
        + (1.0 - y_true) * np.log(1.0 - y_pred)
    )

    return round(float(np.mean(losses)), 4)