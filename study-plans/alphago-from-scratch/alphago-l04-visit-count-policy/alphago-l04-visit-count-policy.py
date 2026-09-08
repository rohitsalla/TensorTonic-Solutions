import numpy as np

def visit_count_policy(visits, legal_mask, temperature, uniform_draw):
    """
    Returns: A tuple of the search policy array and selected action index.
    """
    N    = np.array(visits,     dtype=np.float64)
    mask = np.array(legal_mask, dtype=bool)
    n    = len(N)

    legal_visits = N[mask]
    policy       = np.zeros(n, dtype=np.float64)

    if temperature == 0:
        # One-hot at lowest-index legal action with maximum visits
        max_v    = legal_visits.max()
        legal_idx = np.where(mask)[0]
        winner   = legal_idx[np.argmax(legal_visits == max_v)]  # first max index
        policy[winner] = 1.0
        return (policy, int(winner))

    # Positive temperature: N^(1/τ), with uniform fallback if all visits are zero
    if legal_visits.sum() == 0:
        policy[mask] = 1.0 / mask.sum()
    else:
        powered       = legal_visits ** (1.0 / temperature)
        policy[mask]  = powered / powered.sum()

    # Strict cumulative sampling: first action where cumsum > draw
    legal_idx  = np.where(mask)[0]
    legal_probs = policy[mask]
    cumsum     = np.cumsum(legal_probs)
    sel_local  = int(np.argmax(cumsum > uniform_draw))  # first True
    action     = int(legal_idx[sel_local])

    return (policy, action)