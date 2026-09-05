import torch
import torch.nn.functional as F

def direct_preference_optimization_loss(policy_chosen, policy_rejected,
                                        reference_chosen, reference_rejected,
                                        beta, label_smoothing=0.0):
    """
    Returns: dictionary containing DPO losses, rewards, and accuracy
    """
    # DPO logit: β[(log π+ - log π-) - (log πref+ - log πref-)]
    z = beta * ((policy_chosen - policy_rejected) -
                (reference_chosen - reference_rejected))

    # Label-smoothed loss: -(1-ε) log σ(z) - ε log σ(-z)
    # Using F.logsigmoid for numerical stability
    per_example_loss = (-(1 - label_smoothing) * F.logsigmoid(z)
                        -     label_smoothing  * F.logsigmoid(-z))

    loss = per_example_loss.mean()

    # Implicit rewards
    chosen_rewards   = beta * (policy_chosen   - reference_chosen)
    rejected_rewards = beta * (policy_rejected - reference_rejected)

    # Fraction where chosen reward strictly exceeds rejected reward
    preference_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return {
        "loss":               loss,
        "per_example_loss":   per_example_loss,
        "chosen_rewards":     chosen_rewards,
        "rejected_rewards":   rejected_rewards,
        "preference_accuracy": preference_accuracy,
    }