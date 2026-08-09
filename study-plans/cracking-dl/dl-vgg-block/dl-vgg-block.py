import numpy as np

def _conv2d_valid(x, W, b):
    """Valid (no padding) 2D convolution with stride 1, followed by ReLU."""
    C_in, H, W_dim = x.shape
    C_out, C_in_w, kH, kW = W.shape

    H_out = H - kH + 1
    W_out = W_dim - kW + 1

    out = np.zeros((C_out, H_out, W_out))

    for co in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                window = x[:, i:i + kH, j:j + kW]
                out[co, i, j] = np.sum(window * W[co]) + b[co]

    return np.maximum(0, out)


def _max_pool2x2(x):
    """Non-overlapping 2x2 max pooling."""
    C, H, W = x.shape
    H_out, W_out = H // 2, W // 2

    out = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                out[c, i, j] = np.max(x[c, 2*i:2*i+2, 2*j:2*j+2])

    return out


def vgg_block(x, W1, b1, W2, b2):
    """
    Simplified VGG block: two 3x3 valid convolutions with ReLU + 2x2 max pooling.
    Args:
        x: Input array, shape (in_ch, H, W)
        W1: Conv1 weights, shape (out_ch, in_ch, 3, 3)
        b1: Conv1 bias, shape (out_ch,)
        W2: Conv2 weights, shape (out_ch, out_ch, 3, 3)
        b2: Conv2 bias, shape (out_ch,)
    Returns: Dict with "conv1", "conv2", "pool" as nested lists rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)

    conv1 = _conv2d_valid(x, W1, b1)
    conv2 = _conv2d_valid(conv1, W2, b2)
    pool = _max_pool2x2(conv2)

    return {
        "conv1": np.round(conv1, 4).tolist(),
        "conv2": np.round(conv2, 4).tolist(),
        "pool": np.round(pool, 4).tolist(),
    }