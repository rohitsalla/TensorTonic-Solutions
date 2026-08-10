import numpy as np

def impute(X, method="mean"):
    """
    Returns a new 2D list with NaN values replaced using
    the specified column-wise method.
    """
    if method not in ("mean", "median"):
        raise ValueError("method must be either 'mean' or 'median'")

    # Create a copy so the input is not modified
    result = np.array(X, dtype=float, copy=True)

    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        valid_values = column[~np.isnan(column)]

        if len(valid_values) == 0:
            fill_value = 0.0
        elif method == "mean":
            fill_value = np.mean(valid_values)
        else:
            fill_value = np.median(valid_values)

        missing = np.isnan(column)
        result[missing, column_index] = fill_value

    result = np.round(result, 4)
    result[result == 0] = 0.0

    return result.tolist()