import numpy as np

def layer_normalization(x, gamma, beta, eps=1e-5, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "mean", "var", "x_hat", and optionally "dx", "dgamma", "dbeta".
    """
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)

    N, D = x.shape
    mean = x.mean(axis=1)
    var = x.var(axis=1)  # biased (population) variance, divide by D

    std = np.sqrt(var + eps)
    x_hat = (x - mean[:, None]) / std[:, None]
    output = gamma[None, :] * x_hat + beta[None, :]

    result = {
        "output": np.round(output, 4).tolist(),
        "mean": np.round(mean, 4).tolist(),
        "var": np.round(var, 4).tolist(),
        "x_hat": np.round(x_hat, 4).tolist(),
    }

    if mode == "backward":
        d_output = np.array(d_output, dtype=float)

        dgamma = np.sum(d_output * x_hat, axis=0)
        dbeta = np.sum(d_output, axis=0)

        dx_hat = d_output * gamma[None, :]

        dvar = np.sum(
            dx_hat * (x - mean[:, None]) * -0.5 * (var + eps)[:, None] ** (-1.5),
            axis=1
        )
        dmean = (
            np.sum(dx_hat * -1.0 / std[:, None], axis=1)
            + dvar * np.sum(-2.0 * (x - mean[:, None]), axis=1) / D
        )

        dx = (
            dx_hat / std[:, None]
            + dvar[:, None] * 2.0 * (x - mean[:, None]) / D
            + dmean[:, None] / D
        )

        result["dx"] = np.round(dx, 4).tolist()
        result["dgamma"] = np.round(dgamma, 4).tolist()
        result["dbeta"] = np.round(dbeta, 4).tolist()

    return result