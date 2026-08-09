import numpy as np

def inception_module(x, W_1x1, b_1x1, W_3x3, b_3x3):
    """
    Returns: Dict with "branch_1x1", "branch_3x3", "output", values rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_1x1 = np.array(W_1x1, dtype=float)
    b_1x1 = np.array(b_1x1, dtype=float)
    W_3x3 = np.array(W_3x3, dtype=float)
    b_3x3 = np.array(b_3x3, dtype=float)

    C_in, H, W_dim = x.shape

    # Branch 1: 1x1 (pointwise) convolution + ReLU
    C1 = W_1x1.shape[0]
    branch_1x1 = np.zeros((C1, H, W_dim))
    for co in range(C1):
        for i in range(H):
            for j in range(W_dim):
                branch_1x1[co, i, j] = np.sum(W_1x1[co] * x[:, i, j]) + b_1x1[co]
    branch_1x1 = np.maximum(0, branch_1x1)

    # Branch 2: 3x3 convolution with same-padding (1-pixel zero pad) + ReLU
    C3 = W_3x3.shape[0]
    pad = 1
    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    branch_3x3 = np.zeros((C3, H, W_dim))
    for co in range(C3):
        for i in range(H):
            for j in range(W_dim):
                window = x_padded[:, i:i + 3, j:j + 3]
                branch_3x3[co, i, j] = np.sum(window * W_3x3[co]) + b_3x3[co]
    branch_3x3 = np.maximum(0, branch_3x3)

    # Concatenate along channel axis
    output = np.concatenate([branch_1x1, branch_3x3], axis=0)

    return {
        "branch_1x1": np.round(branch_1x1, 4).tolist(),
        "branch_3x3": np.round(branch_3x3, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }