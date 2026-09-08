import numpy as np

def initialize_mcts_edges(policy_logits, legal_mask):
    """
    Returns: A dictionary containing aligned N, W, Q, and P arrays.
    """
    logits = np.array(policy_logits, dtype=np.float64)
    mask   = np.array(legal_mask,   dtype=bool)
    n      = len(logits)

    # Max-shifted softmax over legal logits only
    legal_logits = logits[mask]
    m            = legal_logits.max()
    exp_vals     = np.exp(legal_logits - m)
    priors_legal = exp_vals / exp_vals.sum()

    # Place legal priors into full-length array; illegal actions stay 0
    P          = np.zeros(n, dtype=np.float64)
    P[mask]    = priors_legal

    return {
        "N": np.zeros(n, dtype=np.int64),
        "W": np.zeros(n, dtype=np.float64),
        "Q": np.zeros(n, dtype=np.float64),
        "P": P,
    }