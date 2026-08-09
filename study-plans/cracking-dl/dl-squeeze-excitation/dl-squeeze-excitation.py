import numpy as np

def squeeze_excitation(x, W1, b1, W2, b2):
    """
    Returns: Dict with "squeezed", "z1", "scale", "output", all rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)

    C, H, W_dim = x.shape

    # Squeeze: global average pooling over spatial dimensions
    squeezed = x.mean(axis=(1, 2))

    # Excitation: two FC layers
    z1 = np.maximum(0, W1 @ squeezed + b1)
    scale = 1.0 / (1.0 + np.exp(-(W2 @ z1 + b2)))

    # Scale: multiply each channel by its scale factor
    output = x * scale[:, None, None]

    return {
        "squeezed": np.round(squeezed, 4).tolist(),
        "z1": np.round(z1, 4).tolist(),
        "scale": np.round(scale, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }