import torch

def grpo_length_normalization_audit(token_log_probs, advantages, token_mask):
    """
    Returns: dictionary containing sequence and token normalization results
    """
    mask_f     = token_mask.float()
    token_sums = (token_log_probs * mask_f).sum(dim=-1)   # (B,)
    n_active   = mask_f.sum(dim=-1)                        # (B,)

    # g_sum = A_i * sum_t(m_it * l_it)
    sequence_summed = advantages * token_sums              # (B,)

    # g_avg = g_sum / n_i  (0 when n_i == 0)
    token_averaged = torch.where(
        n_active > 0,
        sequence_summed / n_active,
        torch.zeros_like(sequence_summed),
    )

    # Magnitude ratio: sum|g_sum| / sum|g_avg|
    sum_abs_sum = sequence_summed.abs().sum()
    sum_abs_avg = token_averaged.abs().sum()

    magnitude_ratio = torch.where(
        sum_abs_avg > 0,
        sum_abs_sum / sum_abs_avg,
        torch.zeros_like(sum_abs_avg),
    )

    return {
        "sequence_summed": sequence_summed,
        "token_averaged":  token_averaged,
        "magnitude_ratio": magnitude_ratio,
    }