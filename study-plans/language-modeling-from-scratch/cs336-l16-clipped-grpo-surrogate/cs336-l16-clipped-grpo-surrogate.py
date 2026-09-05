import torch

def clipped_grpo_surrogate(new_log_probs, old_log_probs, advantages,
                           token_mask, clip_epsilon=0.2):
    """
    Returns: dictionary containing GRPO loss and clipping diagnostics
    """
    # ── Probability ratios ────────────────────────────────────────────────
    log_ratio = torch.clamp(new_log_probs - old_log_probs, -60.0, 60.0)
    r         = torch.exp(log_ratio)                              # (B, T)

    # ── Surrogate terms ───────────────────────────────────────────────────
    A = advantages.unsqueeze(-1)                                  # (B, 1)
    s_unclipped = r * A
    s_clipped   = torch.clamp(r, 1 - clip_epsilon, 1 + clip_epsilon) * A
    s           = torch.minimum(s_unclipped, s_clipped)          # (B, T)

    # ── Mask-aware reductions ─────────────────────────────────────────────
    mask_f = token_mask.float()

    # Per-sequence active-token mean surrogate
    token_counts = mask_f.sum(dim=-1)                             # (B,)
    seq_sum      = (s * mask_f).sum(dim=-1)                      # (B,)
    per_sequence_surrogate = torch.where(
        token_counts > 0,
        seq_sum / token_counts,
        torch.zeros_like(seq_sum),
    )

    # Scalar loss: negative mean over all active tokens
    total_active = token_counts.sum()
    if total_active > 0:
        loss = -(s * mask_f).sum() / total_active
    else:
        loss = torch.zeros(1, dtype=new_log_probs.dtype, device=new_log_probs.device).squeeze()

    # Clipped fraction: ratio strictly outside [1-ε, 1+ε]
    is_clipped    = (r < 1 - clip_epsilon) | (r > 1 + clip_epsilon)
    active_clipped = (is_clipped & token_mask).float().sum()
    clipped_fraction = (active_clipped / total_active) if total_active > 0 else \
                       torch.zeros(1, dtype=new_log_probs.dtype, device=new_log_probs.device).squeeze()

    return {
        "loss":                  loss,
        "per_sequence_surrogate": per_sequence_surrogate,
        "clipped_fraction":      clipped_fraction,
    }