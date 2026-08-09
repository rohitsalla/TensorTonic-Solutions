import numpy as np

def conv2d(input, filters, bias=None, padding=0, stride=1):
    """
    Returns: 3D list of shape (C_out, H_out, W_out) with values rounded to 4 decimal places.
    """
    X = np.array(input, dtype=float)
    F = np.array(filters, dtype=float)

    C_in, H, W = X.shape
    C_out, C_in_f, kH, kW = F.shape

    if bias is None:
        b = np.zeros(C_out)
    else:
        b = np.array(bias, dtype=float)

    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        X_padded = X

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    output = np.zeros((C_out, H_out, W_out))

    for co in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                row_start = i * stride
                col_start = j * stride
                window = X_padded[:, row_start:row_start + kH, col_start:col_start + kW]
                output[co, i, j] = np.sum(window * F[co]) + b[co]

    return np.round(output, 4).tolist()