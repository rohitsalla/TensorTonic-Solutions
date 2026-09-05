import torch
import math

def wsd_learning_rate(total_steps, warmup_steps, stable_steps,
                      peak_learning_rate, final_ratio):
    """
    Returns: float64 learning-rate tensor
    """
    eta_max = peak_learning_rate
    r       = final_ratio
    d       = total_steps - warmup_steps - stable_steps   # decay steps

    rates = []

    # ── Warmup: linearly ramp from eta_max/w to eta_max ──────────────────
    for t in range(warmup_steps):
        rates.append(eta_max * (t + 1) / warmup_steps)

    # ── Stable: hold at peak ──────────────────────────────────────────────
    for _ in range(stable_steps):
        rates.append(eta_max)

    # ── Decay: cosine from peak to eta_max * r ────────────────────────────
    for j in range(d):
        p_j = 1.0 if d == 1 else j / (d - 1)
        eta_j = eta_max * (r + (1 - r) * (1 + math.cos(math.pi * p_j)) / 2)
        rates.append(eta_j)

    return torch.tensor(rates, dtype=torch.float64)