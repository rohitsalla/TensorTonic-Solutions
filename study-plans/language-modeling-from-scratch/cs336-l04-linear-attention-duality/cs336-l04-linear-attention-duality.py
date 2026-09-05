import torch

def linear_attention_duality(q, k, v):
    """
    Returns: dictionary containing both outputs and the final state
    """
    B, S, Dk = q.shape
    Dv = v.shape[2]

    q32 = q.float()
    k32 = k.float()
    v32 = v.float()

    # ── Parallel path ─────────────────────────────────────────────────────
    # Causal (lower-triangular) query-key score matrix: L[b,i,j] = q_i · k_j
    scores = torch.einsum("bid,bjd->bij", q32, k32)           # (B, S, S)
    mask   = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    scores = scores * mask                                     # zero upper triangle

    # Weighted sum over value positions: y_i = sum_{j<=i} L[i,j] * v_j
    parallel_out = torch.einsum("bij,bjd->bid", scores, v32)  # (B, S, Dv)

    # ── Recurrent path ────────────────────────────────────────────────────
    state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=q.device)
    recurrent_out = torch.zeros(B, S, Dv, dtype=torch.float32, device=q.device)

    for t in range(S):
        # S_t = S_{t-1} + k_t v_t^T   outer product accumulated into state
        state = state + torch.einsum("bd,be->bde", k32[:, t], v32[:, t])
        # y_t = q_t^T S_t
        recurrent_out[:, t] = torch.einsum("bd,bde->be", q32[:, t], state)

    # final_state is the state after all S steps
    final_state = state

    return {
        "parallel_output":  parallel_out.to(dtype=q.dtype),
        "recurrent_output": recurrent_out.to(dtype=q.dtype),
        "final_state":      final_state.to(dtype=q.dtype),
    }