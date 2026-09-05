import math
import torch
import torch.nn.functional as F

def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    """
    Returns: dictionary containing output, hidden_width, and parameter_count
    """
    d     = x.shape[-1]
    h_max = w_g.shape[1]

    # Nearest-integer rounding toward budget, clamped to available width
    h = min(h_max, max(1, math.floor(base_params / (3 * d) + 0.5)))

    # Slice to selected hidden width (cols for in-projections, rows for out-projection)
    w_g_h = w_g[:, :h]   # (d, h)
    w_v_h = w_v[:, :h]   # (d, h)
    w_o_h = w_o[:h, :]   # (h, d)

    # Compute in float32, then cast back to input dtype
    x32 = x.float()
    gate   = F.silu(x32 @ w_g_h.float())  # (*, h)
    value  = x32 @ w_v_h.float()          # (*, h)
    output = (gate * value) @ w_o_h.float()  # (*, d)

    return {
        "output":          output.to(dtype=x.dtype),
        "hidden_width":    h,
        "parameter_count": 3 * d * h,
    }