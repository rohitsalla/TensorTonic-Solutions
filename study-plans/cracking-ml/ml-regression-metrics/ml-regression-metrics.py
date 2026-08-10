import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Returns a dict with keys "mse", "mae", and "r2",
    rounded to 4 decimal places.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred

    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))

    residual_sum_squares = np.sum(residuals**2)
    total_sum_squares = np.sum(
        (y_true - np.mean(y_true))**2
    )

    if total_sum_squares == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (
            residual_sum_squares / total_sum_squares
        )

    return {
        "mse": round(float(mse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4)
    }