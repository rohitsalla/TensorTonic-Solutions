import numpy as np

def transposed_conv2d(input, filters, bias=None, stride=1, padding=0):
    """
    Returns: 3D list of shape (C_out, H_out, W_out) with values rounded to 4 decimal places.
    """
    X = np.array(input, dtype=float)
    F = np.array(filters, dtype=float)

    C_in, H_in, W_in = X.shape
    C_in_f, C_out, kH, kW = F.shape

    if bias is None:
        b = np.zeros(C_out)
    else:
        b = np.array(bias, dtype=float)

    H_out = (H_in - 1) * stride - 2 * padding + kH
    W_out = (W_in - 1) * stride - 2 * padding + kW

    output = np.zeros((C_out, H_out, W_out))

    for ci in range(C_in):
        for i_p in range(H_in):
            for j_p in range(W_in):
                val = X[ci, i_p, j_p]
                for co in range(C_out):
                    for m in range(kH):
                        for n in range(kW):
                            i = i_p * stride - padding + m
                            j = j_p * stride - padding + n
                            if 0 <= i < H_out and 0 <= j < W_out:
                                output[co, i, j] += val * F[ci, co, m, n]

    for co in range(C_out):
        output[co] += b[co]

    return np.round(output, 4).tolist()