import numpy as np

def gan_step(real_data, z, W_g, b_g, W_d, b_d):
    """
    Returns: Dict with "fake_data", "real_score", "fake_score", "d_loss", "g_loss", all rounded to 4 decimals.
    """
    real_data = np.array(real_data, dtype=float)
    z = np.array(z, dtype=float)
    W_g = np.array(W_g, dtype=float)
    b_g = np.array(b_g, dtype=float)
    W_d = np.array(W_d, dtype=float)
    b_d = np.array(b_d, dtype=float)

    eps = 1e-7

    # Generator: fake_data_i = tanh(W_g @ z_i + b_g)
    fake_data = np.tanh(z @ W_g.T + b_g)

    # Discriminator: score_i = sigmoid(W_d @ x_i + b_d), squeezed to scalar
    def discriminator(X):
        logits = X @ W_d.T + b_d  # shape (N, 1)
        scores = 1.0 / (1.0 + np.exp(-logits))
        return scores.flatten()

    real_score = discriminator(real_data)
    fake_score = discriminator(fake_data)

    d_loss = -np.mean(np.log(real_score + eps) + np.log(1 - fake_score + eps))
    g_loss = -np.mean(np.log(fake_score + eps))

    return {
        "fake_data": np.round(fake_data, 4).tolist(),
        "real_score": np.round(real_score, 4).tolist(),
        "fake_score": np.round(fake_score, 4).tolist(),
        "d_loss": round(float(d_loss), 4),
        "g_loss": round(float(g_loss), 4),
    }