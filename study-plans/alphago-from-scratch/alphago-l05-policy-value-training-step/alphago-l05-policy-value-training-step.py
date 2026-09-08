import torch
import torch.nn.functional as F

def policy_value_training_step(features, policy_weight, policy_bias, value_weight, value_bias,
                                target_policies, target_values, learning_rate, l2_coefficient):
    """
    Returns: pre-update loss and all four simultaneously updated head parameters.
    """
    # Work in float32; enable grad tracking on parameters only
    X  = features.float().detach()
    pw = policy_weight.float().detach().requires_grad_(True)
    pb = policy_bias.float().detach().requires_grad_(True)
    vw = value_weight.float().detach().requires_grad_(True)
    vb = value_bias.float().detach().requires_grad_(True)

    tp = target_policies.float().detach()
    tv = target_values.float().detach().squeeze(-1)   # (B,)

    # ── Forward pass ──────────────────────────────────────────────────────
    # Policy head: (B, D) @ (A, D)^T + (A,) -> (B, A)
    logits = X @ pw.T + pb
    log_p  = F.log_softmax(logits, dim=-1)
    policy_loss = -(tp * log_p).sum(dim=-1).mean()

    # Value head: tanh(X @ vw^T + vb) -> (B,)
    v_pred      = torch.tanh((X @ vw.unsqueeze(-1)).squeeze(-1) + vb)
    value_loss  = ((v_pred - tv) ** 2).mean()

    # L2 regularization over all four parameters
    reg_loss = l2_coefficient * (
        (pw ** 2).sum() + (pb ** 2).sum() +
        (vw ** 2).sum() + (vb ** 2).sum()
    )

    loss = policy_loss + value_loss + reg_loss

    # ── Compute all gradients from the pre-update loss ────────────────────
    loss.backward()

    # ── Simultaneous SGD update (detach from graph) ───────────────────────
    lr = float(learning_rate)
    with torch.no_grad():
        new_pw = pw - lr * pw.grad
        new_pb = pb - lr * pb.grad
        new_vw = vw - lr * vw.grad
        new_vb = vb - lr * vb.grad

    return (loss.detach(),
            new_pw.detach(), new_pb.detach(),
            new_vw.detach(), new_vb.detach())