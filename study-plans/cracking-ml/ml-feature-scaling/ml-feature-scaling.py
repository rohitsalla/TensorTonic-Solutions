import numpy as np

def feature_scale(X, method="minmax"):
    """
    Returns a new 2D list of scaled values.
    """
    if method not in ("minmax", "standard"):
        raise ValueError(
            "method must be either 'minmax' or 'standard'"
        )

    X = np.asarray(X, dtype=float)

    if method == "minmax":
        column_min = np.min(X, axis=0)
        column_max = np.max(X, axis=0)

        denominator = column_max - column_min
        denominator = np.where(
            denominator == 0,
            1.0,
            denominator
        )

        scaled = (X - column_min) / denominator

    else:
        column_mean = np.mean(X, axis=0)
        column_std = np.std(X, axis=0, ddof=0)

        denominator = np.where(
            column_std == 0,
            1.0,
            column_std
        )

        scaled = (X - column_mean) / denominator

    scaled = np.round(scaled, 4)
    scaled[scaled == 0] = 0.0

    return scaled.tolist()