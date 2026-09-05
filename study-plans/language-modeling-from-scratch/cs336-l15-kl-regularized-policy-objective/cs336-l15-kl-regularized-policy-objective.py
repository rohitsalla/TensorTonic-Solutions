import torch

def kl_regularized_policy_objective(policy_log_probs, reference_log_probs,
                                    rewards, response_mask, beta):
    """
    Returns: dictionary containing objective, penalized rewards, and divergence
    """
    # Sampled KL divergence per sample: sum of masked log-prob differences
    log_ratio = policy_log_probs - reference_log_probs      # (B, T)
    divergences = (log_ratio * response_mask.float()).sum(dim=-1)  # (B,)

    # KL-penalized rewards and mean objective
    penalized_rewards = rewards - beta * divergences         # (B,)
    
    if penalized_rewards.numel() == 0:
        objective = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    else:
        objective = penalized_rewards.mean()                 # scalar

    return {
        "objective":          objective.float(),
        "penalized_rewards":  penalized_rewards.float(),
        "sampled_divergences": divergences.float(),
    }