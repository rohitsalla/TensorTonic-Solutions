import torch

def judge_length_debiasing(num_models, model_a, model_b,
                            length_differences, outcomes,
                            steps, learning_rate):
    """
    Returns: dictionary containing fitted bias and debiased probabilities
    """
    device = outcomes.device
    a      = model_a.long()
    b      = model_b.long()
    delta  = length_differences.double()
    y      = outcomes.double()
    M      = len(y)

    u    = torch.zeros(num_models, dtype=torch.float64, device=device)
    beta = torch.zeros(1,          dtype=torch.float64, device=device)

    for _ in range(steps):
        logits    = u[a] - u[b] + beta * delta          # (M,)
        p         = torch.sigmoid(logits)
        residuals = y - p                                # (M,)

        # Utility gradients via scatter
        g_u = torch.zeros(num_models, dtype=torch.float64, device=device)
        if M > 0:
            g_u.index_add_(0, a,  residuals)
            g_u.index_add_(0, b, -residuals)
            g_u = g_u / M

        # Length-coefficient gradient
        g_beta = (residuals * delta).sum() / M if M > 0 else torch.zeros(1, dtype=torch.float64, device=device)

        u    = u + learning_rate * g_u
        u    = u - u.mean()
        beta = beta + learning_rate * g_beta

    # ── Debiased probabilities (no length term) ───────────────────────────
    p0 = torch.sigmoid(u[a] - u[b])                     # (M,)

    # ── Per-model win rates ───────────────────────────────────────────────
    win_rates = torch.full((num_models,), 0.5, dtype=torch.float64, device=device)

    for i in range(num_models):
        probs = []
        mask_a = (a == i)
        mask_b = (b == i)
        if mask_a.any():
            probs.append(p0[mask_a])
        if mask_b.any():
            probs.append(1.0 - p0[mask_b])
        if probs:
            win_rates[i] = torch.cat(probs).mean()

    return {
        "length_coefficient":    float(beta.item()),
        "debiased_probabilities": p0,
        "model_win_rates":       win_rates,
    }