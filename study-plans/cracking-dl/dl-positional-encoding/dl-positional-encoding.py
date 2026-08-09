import numpy as np

def rope(x, mode="forward", d_output=None):
    """
    Returns: Dict with "rotated", "cos_pe", "sin_pe" (and "dx" in backward mode).
    All values rounded to 4 decimal places.
    """
    x = np.array(x, dtype=float)
    T, d = x.shape
    half = d // 2

    positions = np.arange(T)[:, None]      # (T, 1)
    i = np.arange(half)[None, :]            # (1, half)
    freqs = 1.0 / (10000 ** (2 * i / d))
    theta = positions * freqs               # (T, half)

    cos_pe = np.cos(theta)
    sin_pe = np.sin(theta)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    rot_even = x_even * cos_pe - x_odd * sin_pe
    rot_odd = x_even * sin_pe + x_odd * cos_pe

    rotated = np.zeros_like(x)
    rotated[:, 0::2] = rot_even
    rotated[:, 1::2] = rot_odd

    result = {
        "rotated": np.round(rotated, 4).tolist(),
        "cos_pe": np.round(cos_pe, 4).tolist(),
        "sin_pe": np.round(sin_pe, 4).tolist(),
    }

    if mode == "backward":
        d_output = np.array(d_output, dtype=float)
        d_even = d_output[:, 0::2]
        d_odd = d_output[:, 1::2]

        # Transpose of a 2D rotation matrix = rotation by -theta,
        # equivalently: swap the sign on the sin terms.
        dx_even = d_even * cos_pe + d_odd * sin_pe
        dx_odd = -d_even * sin_pe + d_odd * cos_pe

        dx = np.zeros_like(x)
        dx[:, 0::2] = dx_even
        dx[:, 1::2] = dx_odd

        result["dx"] = np.round(dx, 4).tolist()

    return result