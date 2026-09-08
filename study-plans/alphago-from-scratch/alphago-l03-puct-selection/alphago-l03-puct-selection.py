import numpy as np

def select_puct_action(visits, total_values, priors, legal_mask, cpuct):
    """
    Returns: The selected action index and the complete PUCT score vector.
    """
    N   = np.array(visits,       dtype=np.float64)
    W   = np.array(total_values, dtype=np.float64)
    P   = np.array(priors,       dtype=np.float64)
    mask = np.array(legal_mask,  dtype=bool)

    # Q: mean value, 0 when unvisited
    Q = np.where(N > 0, W / N, 0.0)

    # T: total visits at this node
    T = N.sum()

    # PUCT score: Q + c * P * sqrt(T) / (1 + N)
    S = Q + cpuct * P * np.sqrt(T) / (1.0 + N)

    # Mask illegal actions with -inf
    S = np.where(mask, S, -np.inf)

    # argmax gives lowest index on tie (numpy returns first occurrence)
    action = int(np.argmax(S))

    return (action, S)