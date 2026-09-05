import torch

def muon_spectral_update(parameter, gradient, previous_momentum,
                         momentum_coefficient, learning_rate):
    """
    Returns: dictionary containing parameter, momentum, and spectral update
    """
    # Momentum update: B_t = μ B_{t-1} + G_t
    new_momentum = momentum_coefficient * previous_momentum + gradient

    # Compact SVD in float32: B_t = U Σ V^T
    B32 = new_momentum.float()
    U, S, Vh = torch.linalg.svd(B32, full_matrices=False)

    # Replace singular values with 1: O_t = U V^T
    orthogonalized = U @ Vh   # (M, N)

    # Parameter update: W_{t+1} = W_t - η O_t
    new_parameter = parameter - learning_rate * orthogonalized.to(dtype=parameter.dtype)

    return {
        "new_parameter":        new_parameter.to(dtype=parameter.dtype),
        "new_momentum":         new_momentum.to(dtype=parameter.dtype),
        "orthogonalized_update": orthogonalized.to(dtype=parameter.dtype),
    }