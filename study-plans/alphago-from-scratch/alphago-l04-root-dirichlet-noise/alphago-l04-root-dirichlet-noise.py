import numpy as np

def mix_root_dirichlet_noise(priors, noise, legal_mask, epsilon):
    """
    Returns: A normalized root-prior array with noise mixed over legal actions.
    """
    P    = np.array(priors,     dtype=np.float64)
    eta  = np.array(noise,      dtype=np.float64)
    mask = np.array(legal_mask, dtype=bool)

    # Normalize priors and noise independently over legal actions only
    P_legal   = P[mask];   P_hat   = P_legal   / P_legal.sum()
    eta_legal = eta[mask]; eta_hat = eta_legal / eta_legal.sum()

    # Mix: (1-ε) * normalized_prior + ε * normalized_noise
    mixed_legal = (1 - epsilon) * P_hat + epsilon * eta_hat

    # Place mixed values at legal positions; illegal actions stay 0
    result       = np.zeros(len(P), dtype=np.float64)
    result[mask] = mixed_legal

    return result