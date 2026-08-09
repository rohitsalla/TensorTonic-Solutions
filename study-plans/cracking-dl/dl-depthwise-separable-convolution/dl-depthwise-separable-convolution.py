import numpy as np

def depthwise_separable_conv(x, W_dw, b_dw, W_pw, b_pw):
    """
    Returns: Dict with "depthwise" and "pointwise", values as nested lists rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_dw = np.array(W_dw, dtype=float)
    b_dw = np.array(b_dw, dtype=float)
    W_pw = np.array(W_pw, dtype=float)
    b_pw = np.array(b_pw, dtype=float)

    C, H, W = x.shape
    kH, kW = W_dw.shape[1], W_dw.shape[2]
    H_out = H - kH + 1
    W_out = W - kW + 1

    # Stage 1: depthwise convolution (each channel filtered independently)
    dw = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                window = x[c, i:i + kH, j:j + kW]
                dw[c, i, j] = np.sum(window * W_dw[c]) + b_dw[c]
    dw = np.maximum(0, dw)

    # Stage 2: pointwise (1x1) convolution across channels
    C_out = W_pw.shape[0]
    pw = np.zeros((C_out, H_out, W_out))
    for co in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                pw[co, i, j] = np.sum(W_pw[co] * dw[:, i, j]) + b_pw[co]
    pw = np.maximum(0, pw)

    return {
        "depthwise": np.round(dw, 4).tolist(),
        "pointwise": np.round(pw, 4).tolist()
    }