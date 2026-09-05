import torch

def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    """
    Returns: dictionary containing new_param and full_grad tensors
    """
    # Total examples across all microbatches
    N = sum(X_m.shape[0] for X_m in microbatch_inputs)

    # Detached clone with grad tracking — never touches the original param or its .grad
    w = param.detach().clone().requires_grad_(True)

    # Accumulate gradients one microbatch at a time
    for X_m, y_m in zip(microbatch_inputs, microbatch_targets):
        X_m = X_m.to(dtype=param.dtype)
        y_m = y_m.to(dtype=param.dtype)

        # Sum squared errors divided by N (not N_m), so each microbatch
        # contributes its exact N_m/N share of the full-batch mean
        residuals = X_m @ w - y_m          # (N_m,)
        loss = (residuals ** 2).sum() / N  # weighted scalar
        loss.backward()                    # accumulates into w.grad

    full_grad = w.grad.detach().clone()
    new_param  = (w - lr * full_grad).detach()

    return {
        "new_param": new_param.to(dtype=param.dtype),
        "full_grad": full_grad.to(dtype=param.dtype),
    }