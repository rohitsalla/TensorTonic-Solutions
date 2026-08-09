import numpy as np

def unet_skip(x, W_down, b_down, W_up, b_up, W_out, b_out):
    """
    Returns: Dict with "encoded", "decoded", "combined", "output", values rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_down = np.array(W_down, dtype=float)
    b_down = np.array(b_down, dtype=float)
    W_up = np.array(W_up, dtype=float)
    b_up = np.array(b_up, dtype=float)
    W_out = np.array(W_out, dtype=float)
    b_out = np.array(b_out, dtype=float)

    encoded = np.maximum(0, W_down @ x + b_down)
    decoded = np.maximum(0, W_up @ encoded + b_up)
    combined = decoded + x
    output = W_out @ combined + b_out

    return {
        "encoded": np.round(encoded, 4).tolist(),
        "decoded": np.round(decoded, 4).tolist(),
        "combined": np.round(combined, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }