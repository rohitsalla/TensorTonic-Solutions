import numpy as np

def vae_forward(x, W_enc, b_enc, W_mu, b_mu, W_logvar, b_logvar, W_dec, b_dec, z_sample):
    """
    Returns: Dict with "hidden", "mu", "log_var", "z", "reconstruction", values rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_enc = np.array(W_enc, dtype=float)
    b_enc = np.array(b_enc, dtype=float)
    W_mu = np.array(W_mu, dtype=float)
    b_mu = np.array(b_mu, dtype=float)
    W_logvar = np.array(W_logvar, dtype=float)
    b_logvar = np.array(b_logvar, dtype=float)
    W_dec = np.array(W_dec, dtype=float)
    b_dec = np.array(b_dec, dtype=float)
    z_sample = np.array(z_sample, dtype=float)

    hidden = np.maximum(0, W_enc @ x + b_enc)
    mu = W_mu @ hidden + b_mu
    log_var = W_logvar @ hidden + b_logvar

    z = mu + np.exp(0.5 * log_var) * z_sample

    reconstruction = 1.0 / (1.0 + np.exp(-(W_dec @ z + b_dec)))

    return {
        "hidden": np.round(hidden, 4).tolist(),
        "mu": np.round(mu, 4).tolist(),
        "log_var": np.round(log_var, 4).tolist(),
        "z": np.round(z, 4).tolist(),
        "reconstruction": np.round(reconstruction, 4).tolist(),
    }