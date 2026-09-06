import torch

def sgd_training_step(inputs, targets, weights, bias, learning_rate):
    """
    Returns: old and new loss, updated parameters, and current-step parameter gradients
    """
    X  = inputs   # (B, D)
    y  = targets  # (B,)
    w  = weights  # (D,)
    b  = bias     # scalar tensor

    # ── Forward pass ──────────────────────────────────────────────────────
    a = X @ w + b                              # (B,) preactivations
    p = torch.tanh(a)                          # (B,) predictions

    # ── Old loss ──────────────────────────────────────────────────────────
    old_loss = ((p - y) ** 2).sum()

    # ── Manual gradients ──────────────────────────────────────────────────
    delta  = 2 * (p - y) * (1 - p * p)        # (B,) preactivation gradients
    grad_w = X.T @ delta                       # (D,) weight gradient
    grad_b = delta.sum()                       # scalar bias gradient

    # ── Parameter update ──────────────────────────────────────────────────
    # Use Python float for lr to avoid any tensor dtype mismatch
    lr     = float(learning_rate)
    new_w  = w - lr * grad_w                   # (D,)
    new_b  = b - lr * grad_b                   # scalar

    # ── New loss ──────────────────────────────────────────────────────────
    p_new    = torch.tanh(X @ new_w + new_b)
    new_loss = ((p_new - y) ** 2).sum()

    return (old_loss, new_loss, new_w, new_b, grad_w, grad_b)