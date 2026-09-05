import torch

def gated_deltanet_scan(q, k, v, gamma, beta):
    """
    Returns: dictionary containing sequence outputs and the final state
    """
    B, S, Dk = q.shape
    Dv = v.shape[2]

    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    g32 = gamma.float()   # (B, S)
    b32 = beta.float()    # (B, S)

    state   = torch.zeros(B, Dk, Dv, device=q.device, dtype=torch.float32)
    outputs = torch.zeros(B, S,  Dv, device=q.device, dtype=torch.float32)

    # Pre-build identity once, expand to batch
    I = torch.eye(Dk, device=q.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    for t in range(S):
        kt = k32[:, t]          # (B, Dk)
        vt = v32[:, t]          # (B, Dv)
        qt = q32[:, t]          # (B, Dk)
        gt = g32[:, t].reshape(B, 1, 1)   # (B, 1, 1) for state broadcast
        bt = b32[:, t].reshape(B, 1, 1)   # (B, 1, 1)

        # Erase matrix: (I - β k k^T), shape (B, Dk, Dk)
        kk_outer = kt.unsqueeze(2) * kt.unsqueeze(1)   # (B, Dk, Dk)
        erase    = I - bt * kk_outer

        # Gated erase + write: S_t = γ (I - β k k^T) S_{t-1} + β k v^T
        kv_outer = kt.unsqueeze(2) * vt.unsqueeze(1)   # (B, Dk, Dv)
        state    = gt * torch.bmm(erase, state) + bt * kv_outer

        # Read: y_t = q_t^T S_t
        outputs[:, t] = torch.bmm(qt.unsqueeze(1), state).squeeze(1)

    return {
        "outputs":     outputs.to(dtype=q.dtype),
        "final_state": state.to(dtype=q.dtype),
    }