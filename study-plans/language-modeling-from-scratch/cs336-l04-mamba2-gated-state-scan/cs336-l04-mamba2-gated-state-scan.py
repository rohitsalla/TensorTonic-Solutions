import torch

def mamba2_gated_scan(q, k, v, gamma):
    """
    Returns: dictionary containing sequence outputs and the final state
    """
    B, S, Dk = q.shape
    Dv = v.shape[2]

    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    g32 = gamma.float()

    state   = torch.zeros(B, Dk, Dv, device=q.device, dtype=torch.float32)
    outputs = torch.zeros(B, S,  Dv, device=q.device, dtype=torch.float32)

    for t in range(S):
        # Decay previous state by scalar gate, then accumulate outer product
        # gamma[:, t]: (B,) -> (B, 1, 1) for broadcasting over (B, Dk, Dv)
        state = g32[:, t].reshape(B, 1, 1) * state \
              + torch.einsum("bd,be->bde", k32[:, t], v32[:, t])

        # Read output from updated state: q_t^T S_t -> (B, 1, Dk) @ (B, Dk, Dv) = (B, 1, Dv)
        outputs[:, t] = torch.bmm(q32[:, t].unsqueeze(1), state).squeeze(1)

    return {
        "outputs":     outputs.to(dtype=q.dtype),
        "final_state": state.to(dtype=q.dtype),
    }