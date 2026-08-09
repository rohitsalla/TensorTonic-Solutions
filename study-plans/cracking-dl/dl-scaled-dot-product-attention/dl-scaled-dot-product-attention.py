import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "attention_weights", and optionally "dQ", "dK", "dV".
    """
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    V = np.array(V, dtype=float)

    dk = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(dk)

    if mask is not None:
        mask = np.array(mask, dtype=bool)
        scores = np.where(mask, scores, -1e9)

    # Numerically stable softmax along the last axis
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = attention_weights @ V

    result = {
        "output": np.round(output, 4).tolist(),
        "attention_weights": np.round(attention_weights, 4).tolist(),
    }

    if mode == "backward":
        d_output = np.array(d_output, dtype=float)

        # dV = W^T @ dOutput
        dV = attention_weights.T @ d_output

        # dW = dOutput @ V^T
        dW = d_output @ V.T

        # Softmax Jacobian (per row): dS = W * (dW - sum(dW * W, axis=-1))
        sum_term = np.sum(dW * attention_weights, axis=-1, keepdims=True)
        dS = attention_weights * (dW - sum_term)

        dQ = (dS @ K) / np.sqrt(dk)
        dK = (dS.T @ Q) / np.sqrt(dk)

        result["dQ"] = np.round(dQ, 4).tolist()
        result["dK"] = np.round(dK, 4).tolist()
        result["dV"] = np.round(dV, 4).tolist()

    return result