import numpy as np

def autoencoder(x, W_enc, b_enc, W_dec, b_dec):
    """
    Returns: Dict with "encoded" and "decoded", values rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_enc = np.array(W_enc, dtype=float)
    b_enc = np.array(b_enc, dtype=float)
    W_dec = np.array(W_dec, dtype=float)
    b_dec = np.array(b_dec, dtype=float)

    # Encoder: z = relu(W_enc @ x + b_enc)
    z = np.maximum(0, W_enc @ x + b_enc)

    # Decoder: x_hat = sigmoid(W_dec @ z + b_dec)
    x_hat = 1.0 / (1.0 + np.exp(-(W_dec @ z + b_dec)))

    return {
        "encoded": np.round(z, 4).tolist(),
        "decoded": np.round(x_hat, 4).tolist(),
    }