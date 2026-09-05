import numpy as np

def temperature_data_mixture(token_counts, alpha, training_budget):
    """
    Returns: dictionary containing probabilities, tokens, and epochs
    """
    n = np.array(token_counts, dtype=np.float64)
    N = len(n)

    probs          = np.zeros(N, dtype=np.float64)
    expected_tokens = np.zeros(N, dtype=np.float64)
    expected_epochs = np.zeros(N, dtype=np.float64)

    pos_mask = n > 0

    if pos_mask.any():
        # Log weights with max-shift for numerical stability
        log_n     = np.log(n[pos_mask])
        log_w     = alpha * log_n
        log_w    -= log_w.max()
        w         = np.exp(log_w)
        w        /= w.sum()

        probs[pos_mask]           = w
        expected_tokens[pos_mask] = training_budget * w
        expected_epochs[pos_mask] = training_budget * w / n[pos_mask]

    return {
        "probabilities":   probs,
        "expected_tokens": expected_tokens,
        "expected_epochs": expected_epochs,
    }