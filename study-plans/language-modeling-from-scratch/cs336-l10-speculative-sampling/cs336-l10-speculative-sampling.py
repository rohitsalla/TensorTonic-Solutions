import torch

def speculative_sampling(draft_tokens, draft_probabilities, target_probabilities,
                         acceptance_random_values, residual_random_value):
    """
    Returns: dictionary containing emitted tokens and acceptance information
    """
    K = len(draft_tokens)
    accepted_tokens = []

    for i in range(K):
        xi    = int(draft_tokens[i])
        q_i   = draft_probabilities[i].double()
        p_i   = target_probabilities[i].double()
        u     = float(acceptance_random_values[i])

        q_xi = float(q_i[xi])
        p_xi = float(p_i[xi])

        # Acceptance probability with edge-case handling
        if q_xi == 0.0 and p_xi == 0.0:
            a = 0.0
        elif q_xi == 0.0:
            a = 1.0
        else:
            a = min(1.0, p_xi / q_xi)

        if u < a:
            # Accept this draft token
            accepted_tokens.append(xi)
        else:
            # Reject: sample replacement from normalized positive residual
            residual = torch.clamp(p_i - q_i, min=0.0)
            residual = residual / residual.sum()

            # Inverse-CDF (lowest-index) sampling
            cdf = torch.cumsum(residual, dim=0)
            replacement = int((cdf <= residual_random_value).sum())
            replacement = min(replacement, len(cdf) - 1)

            tokens_tensor = torch.tensor(
                accepted_tokens + [replacement],
                dtype=torch.int64,
                device=draft_tokens.device,
            )
            return {
                "tokens":         tokens_tensor,
                "accepted_count": len(accepted_tokens),
                "rejected_at":    i,
            }

    # All K tokens accepted
    tokens_tensor = torch.tensor(
        accepted_tokens,
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    return {
        "tokens":         tokens_tensor,
        "accepted_count": K,
        "rejected_at":    None,
    }