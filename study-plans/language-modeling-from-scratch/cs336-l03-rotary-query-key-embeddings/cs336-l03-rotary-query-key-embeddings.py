import torch

def rotary_embed(q, k, positions, inv_freq):
    """
    Returns: dictionary containing q_rotated and k_rotated
    """
    # Reshape positions for broadcasting: (1,1,S,1) or (B,1,S,1)
    pos = positions.float()
    if pos.dim() == 1:
        pos = pos.reshape(1, 1, -1, 1)       # shared across batch
    else:
        pos = pos.reshape(pos.shape[0], 1, pos.shape[1], 1)  # per-batch

    # inv_freq: (D/2,) -> (1, 1, 1, D/2)
    freq = inv_freq.float().reshape(1, 1, 1, -1)

    # Angles: (1|B, 1, S, D/2) — broadcasts over H when used below
    theta = pos * freq
    cos_t = theta.cos()
    sin_t = theta.sin()

    def rotate(t):
        t32     = t.float()
        x_even  = t32[..., 0::2]                           # (B, H, S, D/2)
        x_odd   = t32[..., 1::2]

        out_even = x_even * cos_t - x_odd  * sin_t        # x2i'
        out_odd  = x_even * sin_t + x_odd  * cos_t        # x2i+1'

        # Interleave: stack -> (B, H, S, D/2, 2), flatten -> (B, H, S, D)
        return torch.stack([out_even, out_odd], dim=-1).flatten(start_dim=-2).to(dtype=t.dtype)

    return {"q_rotated": rotate(q), "k_rotated": rotate(k)}
