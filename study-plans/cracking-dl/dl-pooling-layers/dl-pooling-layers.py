import numpy as np

def pooling(input, pool_size, stride, pool_type):
    """
    Returns: 3D list with pooled values rounded to 4 decimal places.
    """
    X = np.array(input, dtype=float)
    C, H, W = X.shape

    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    output = np.zeros((C, out_h, out_w))

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                row_start = i * stride
                col_start = j * stride
                window = X[c, row_start:row_start + pool_size, col_start:col_start + pool_size]

                if pool_type == "max":
                    output[c, i, j] = np.max(window)
                elif pool_type == "average":
                    output[c, i, j] = np.mean(window)
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")

    return np.round(output, 4).tolist()