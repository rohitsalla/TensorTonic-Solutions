import torch
import torch.nn.functional as F

def alpha_go_policy_value_loss(policy_logits, predicted_values, target_policies,
                               target_values, parameters, l2_coefficient):
    """
    Returns: Total, policy, value, and L2 regularization scalar tensors.
    """
    # ── Policy loss: -mean(Σ π * log_softmax(logits)) ────────────────────
    log_p        = F.log_softmax(policy_logits.float(), dim=-1)   # (B, A)
    policy_loss  = -(target_policies.float() * log_p).sum(dim=-1).mean()

    # ── Value loss: mean((v - z)²) ────────────────────────────────────────
    v           = predicted_values.float().squeeze(-1)
    z           = target_values.float().squeeze(-1)
    value_loss  = ((v - z) ** 2).mean()

    # ── L2 regularization: λ * Σ_j ||θ_j||² ─────────────────────────────
    if len(parameters) == 0 or l2_coefficient == 0:
        reg_loss = torch.tensor(0.0, dtype=torch.float32,
                                device=policy_logits.device)
    else:
        reg_loss = l2_coefficient * sum(
            (p.float() ** 2).sum() for p in parameters
        )
        reg_loss = reg_loss.float()

    total_loss = policy_loss + value_loss + reg_loss

    return (total_loss, policy_loss, value_loss, reg_loss)