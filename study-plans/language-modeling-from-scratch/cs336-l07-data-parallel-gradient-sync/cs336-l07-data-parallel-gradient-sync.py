import torch

def data_parallel_gradient_sync(parameter_replicas, gradient_replicas, learning_rate):
    """
    Returns: dictionary containing mean gradients and updated replicas
    """
    R        = len(parameter_replicas)
    n_params = len(parameter_replicas[0])

    def to_f32(t): return t.float()

    # ── Compute mean gradient per parameter ──────────────────────────────
    mean_gradients = []
    for j in range(n_params):
        param = parameter_replicas[0][j]           # reference shape/dtype/device

        # Sum gradients across ranks (None counts as zero)
        grad_sum = torch.zeros_like(param, dtype=torch.float32)
        for r in range(R):
            g = gradient_replicas[r][j]
            if g is not None:
                grad_sum = grad_sum + to_f32(g)

        mean_grad = (grad_sum / R).to(dtype=param.dtype)
        mean_gradients.append(mean_grad)

    # ── Apply SGD update to each parameter ───────────────────────────────
    updated_params = []
    for j in range(n_params):
        param      = parameter_replicas[0][j]
        mean_grad  = mean_gradients[j]
        updated    = (to_f32(param) - learning_rate * to_f32(mean_grad)).to(dtype=param.dtype)
        updated_params.append(updated)

    # ── Give every rank an independent copy of the updated parameters ─────
    updated_replicas = [
        [p.clone() for p in updated_params]
        for _ in range(R)
    ]

    return {
        "mean_gradients": mean_gradients,
        "updated_replicas": updated_replicas,
    }