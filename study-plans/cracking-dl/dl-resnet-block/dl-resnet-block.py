import numpy as np

def _conv2d_same(x, W, b):
    """3x3 convolution with 1-pixel zero-padding, stride 1 (output spatial size = input)."""
    C_in, H, W_dim = x.shape
    C_out, C_in_w, kH, kW = W.shape
    pad = 1

    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    out = np.zeros((C_out, H, W_dim))
    for co in range(C_out):
        for i in range(H):
            for j in range(W_dim):
                window = x_padded[:, i:i + kH, j:j + kW]
                out[co, i, j] = np.sum(window * W[co]) + b[co]

    return out


def resnet_block(x, W1, b1, W2, b2):
    """
    Returns: dict with "conv1", "conv2", "output" as nested lists rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)

    h1 = np.maximum(0, _conv2d_same(x, W1, b1))
    h2 = _conv2d_same(h1, W2, b2)
    output = np.maximum(0, h2 + x)

    return {
        "conv1": np.round(h1, 4).tolist(),
        "conv2": np.round(h2, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }