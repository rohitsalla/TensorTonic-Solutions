import numpy as np

def _conv2d_valid(x, W, b):
    """Valid (no padding) 2D convolution, stride 1, followed by ReLU. x: (1,H,W), W: (F,1,kH,kW)"""
    C_in, H, W_dim = x.shape
    F, C_in_w, kH, kW = W.shape

    H_out = H - kH + 1
    W_out = W_dim - kW + 1

    out = np.zeros((F, H_out, W_out))
    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):
                window = x[:, i:i + kH, j:j + kW]
                out[f, i, j] = np.sum(window * W[f]) + b[f]

    return np.maximum(0, out)


def _max_pool2x2(x):
    """Non-overlapping 2x2 max pooling. x: (C,H,W)"""
    C, H, W = x.shape
    H_out, W_out = H // 2, W // 2

    out = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                out[c, i, j] = np.max(x[c, 2*i:2*i+2, 2*j:2*j+2])

    return out


def lenet_forward(x, W_conv, b_conv, W_fc1, b_fc1, W_fc2, b_fc2):
    """
    Simplified LeNet forward pass.
    Returns: Dict with "conv_out", "pool_out", "fc1_out", "logits", all rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_conv = np.array(W_conv, dtype=float)
    b_conv = np.array(b_conv, dtype=float)
    W_fc1 = np.array(W_fc1, dtype=float)
    b_fc1 = np.array(b_fc1, dtype=float)
    W_fc2 = np.array(W_fc2, dtype=float)
    b_fc2 = np.array(b_fc2, dtype=float)

    # 1. Convolution + ReLU
    conv_out = _conv2d_valid(x, W_conv, b_conv)

    # 2. Max pool 2x2
    pool_out = _max_pool2x2(conv_out)

    # 3. Flatten (channel-major, row-major within each channel)
    flat = pool_out.flatten()

    # 4. FC1 + ReLU
    fc1_out = np.maximum(0, W_fc1 @ flat + b_fc1)

    # 5. FC2 (raw logits)
    logits = W_fc2 @ fc1_out + b_fc2

    return {
        "conv_out": np.round(conv_out, 4).tolist(),
        "pool_out": np.round(pool_out, 4).tolist(),
        "fc1_out": np.round(fc1_out, 4).tolist(),
        "logits": np.round(logits, 4).tolist(),
    }