import numpy as np

def batch_norm(X, gamma, beta, running_mean, running_var, mode):
    """
    Returns: dict with keys "output", "running_mean", "running_var"
    """
    X = np.array(X, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    running_mean = np.array(running_mean, dtype=float)
    running_var = np.array(running_var, dtype=float)

    momentum = 0.1
    eps = 1e-5

    if mode == "train":
        mu = X.mean(axis=0)
        var = X.var(axis=0)  # population variance (ddof=0)

        x_hat = (X - mu) / np.sqrt(var + eps)
        output = gamma * x_hat + beta

        new_running_mean = (1 - momentum) * running_mean + momentum * mu
        new_running_var = (1 - momentum) * running_var + momentum * var

    elif mode == "test":
        x_hat = (X - running_mean) / np.sqrt(running_var + eps)
        output = gamma * x_hat + beta

        new_running_mean = running_mean
        new_running_var = running_var

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "output": np.round(output, 4).tolist(),
        "running_mean": np.round(new_running_mean, 4).tolist(),
        "running_var": np.round(new_running_var, 4).tolist()
    }