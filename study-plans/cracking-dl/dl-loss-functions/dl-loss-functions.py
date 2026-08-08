import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    if loss_type == "mse":
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        loss = np.mean((y_true - y_pred) ** 2)

    elif loss_type == "bce":
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    elif loss_type == "cce":
        y_true = np.array(y_true, dtype=int)
        logits = np.array(y_pred, dtype=float)
        n = logits.shape[0]
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1)) + max_logits.flatten()
        correct_class_logits = logits[np.arange(n), y_true]
        losses = -(correct_class_logits - log_sum_exp)
        loss = np.mean(losses)

    elif loss_type == "hinge":
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        loss = np.mean(np.maximum(0, 1 - y_true * y_pred))

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return round(float(loss), 4)