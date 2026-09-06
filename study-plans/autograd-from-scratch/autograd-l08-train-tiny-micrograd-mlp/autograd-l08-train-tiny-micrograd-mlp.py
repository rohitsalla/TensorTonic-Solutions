import torch

def train_tiny_micrograd_mlp(inputs, targets, weights, biases, learning_rate, steps):
    """
    Returns: final predictions, final loss, trained parameters, and pre-update loss history
    """
    lr = float(learning_rate)

    # Work with mutable copies of parameters (don't modify inputs)
    Ws = [W.clone() for W in weights]
    bs = [b.clone() for b in biases]

    X = inputs                          # (B, D_in)
    y = targets.unsqueeze(1)            # (B, 1) — treat as batch-by-one column

    loss_history = []

    for _ in range(steps):
        # ── Forward pass: save all activations ───────────────────────────
        H = [X]   # H[0] = input, H[l] = activation after layer l
        for W, b in zip(Ws, bs):
            # H_{l-1} @ W_l^T + b_l  -> (B, out_width)
            H.append(torch.tanh(H[-1] @ W.T + b))

        # ── Loss (pre-update) ─────────────────────────────────────────────
        preds = H[-1]                   # (B, 1)
        loss  = ((preds - y) ** 2).sum()
        loss_history.append(loss)

        # ── Backward pass: compute all gradients before any update ────────
        # Seed: G_L = 2(H_L - y)
        G = 2 * (H[-1] - y)            # (B, 1)

        grad_Ws = []
        grad_bs = []

        for l in reversed(range(len(Ws))):
            # Preactivation gradient: D_l = G_l ⊙ (1 - H_l²)
            D = G * (1 - H[l+1] ** 2)  # (B, out_width)

            # Weight gradient: D_l^T @ H_{l-1}  -> (out_width, in_width)
            grad_Ws.insert(0, D.T @ H[l])

            # Bias gradient: sum over batch  -> (out_width,)
            grad_bs.insert(0, D.sum(dim=0))

            # Pass gradient to preceding layer: G_{l-1} = D_l @ W_l
            G = D @ Ws[l]              # (B, in_width)

        # ── Simultaneous parameter update ─────────────────────────────────
        for l in range(len(Ws)):
            Ws[l] = Ws[l] - lr * grad_Ws[l]
            bs[l] = bs[l] - lr * grad_bs[l]

    # ── Final forward pass after last update ──────────────────────────────
    H = X
    for W, b in zip(Ws, bs):
        H = torch.tanh(H @ W.T + b)

    final_preds = H.squeeze(1)         # (B,)
    final_loss  = ((H - y) ** 2).sum()

    return (final_preds, final_loss, Ws, bs, loss_history)