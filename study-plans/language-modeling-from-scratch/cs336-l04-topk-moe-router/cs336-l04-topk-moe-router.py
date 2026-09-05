import torch

def topk_moe_router(x, w_router, expert_outputs, shared_output, k):
    """
    Returns: dictionary containing routed output, selections, weights, and load statistics
    """
    T, D  = x.shape
    E     = w_router.shape[1]
    Dout  = expert_outputs.shape[2]

    x32   = x.float()
    eo32  = expert_outputs.float()
    sh32  = shared_output.float()

    # ── Router scores: (T, E) ────────────────────────────────────────────
    scores = x32 @ w_router.float()                     # (T, E)

    # ── Top-k selection: stable descending sort keeps lower index on ties ─
    # argsort ascending of negated scores, stable=True -> lower index wins ties
    order           = torch.argsort(-scores, dim=-1, stable=True)  # (T, E)
    selected_experts = order[:, :k].contiguous()                   # (T, k)  long

    # ── Gather selected scores and softmax over them ──────────────────────
    sel_scores   = scores.gather(1, selected_experts)              # (T, k)
    sel_weights  = torch.softmax(sel_scores, dim=-1)               # (T, k)  float32

    # ── Gather expert outputs for selected experts ────────────────────────
    # expert_outputs: (T, E, Dout) -> need (T, k, Dout)
    idx_expanded = selected_experts.unsqueeze(-1).expand(T, k, Dout)  # (T, k, Dout)
    sel_outputs  = eo32.gather(1, idx_expanded)                    # (T, k, Dout)

    # ── Weighted combination + shared residual ────────────────────────────
    weighted = (sel_weights.unsqueeze(-1) * sel_outputs).sum(dim=1)  # (T, Dout)
    output   = sh32 + weighted                                     # (T, Dout)

    # ── Load statistics ───────────────────────────────────────────────────
    # token_fraction: fraction of tokens that selected each expert
    # Build (T, E) boolean selection mask via scatter
    mask = torch.zeros(T, E, dtype=torch.float32, device=x.device)
    mask.scatter_(1, selected_experts, 1.0)
    token_fraction   = mask.mean(dim=0)                            # (E,)

    # probability_mass: mean full softmax probability across tokens
    probability_mass = torch.softmax(scores, dim=-1).mean(dim=0)   # (E,)

    return {
        "output":           output.to(dtype=x.dtype),
        "selected_experts": selected_experts,                       # (T, k) long
        "selected_weights": sel_weights.to(dtype=x.dtype),
        "token_fraction":   token_fraction.to(dtype=x.dtype),
        "probability_mass": probability_mass.to(dtype=x.dtype),
    }