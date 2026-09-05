import torch

def grpo_group_advantages(rewards, group_ids, epsilon=1e-8):
    """
    Returns: dictionary containing normalized advantages and active groups
    """
    if rewards.numel() == 0:
        return {
            "advantages":        rewards.clone(),
            "active_group_mask": torch.zeros(0, dtype=torch.bool, device=rewards.device),
            "active_group_count": 0,
        }

    num_groups = int(group_ids.max().item()) + 1
    advantages = torch.zeros_like(rewards)

    active_group_mask = torch.zeros(num_groups, dtype=torch.bool, device=rewards.device)

    for g in range(num_groups):
        mask      = group_ids == g
        group_r   = rewards[mask]

        mu        = group_r.mean()
        std       = group_r.std(correction=0)   # population std

        if std > 0:
            advantages[mask]    = (group_r - mu) / (std + epsilon)
            active_group_mask[g] = True
        # else: constant group — advantages stay 0, mask stays False

    return {
        "advantages":         advantages,
        "active_group_mask":  active_group_mask,
        "active_group_count": int(active_group_mask.sum().item()),
    }