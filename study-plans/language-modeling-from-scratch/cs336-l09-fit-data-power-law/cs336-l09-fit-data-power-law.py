import torch
import math

def fit_data_power_law(data_sizes, losses, target_sizes, floor_candidates):
    """
    Returns: dictionary containing the selected fit and predictions
    """
    D = data_sizes.double()       # (N,) float64
    L = losses.double()           # (N,)
    T = target_sizes.double()     # (M,)

    log_D = torch.log(D)
    N     = len(D)

    # Design matrix columns shared across all floor fits
    ones = torch.ones(N, dtype=torch.float64, device=D.device)
    X    = torch.stack([ones, log_D], dim=1)   # (N, 2)
    XtX  = X.T @ X                              # (2, 2)

    best_rmse = float('inf')
    best      = None

    for E_t in floor_candidates:
        E = float(E_t)

        # Floor must be strictly below every observed loss
        if E >= float(L.min()):
            continue

        residuals = L - E
        if not (residuals > 0).all():
            continue

        log_r = torch.log(residuals)       # (N,)

        # OLS in log space: [a, b] = (X^T X)^{-1} X^T log_r
        Xty  = X.T @ log_r                # (2,)
        beta = torch.linalg.solve(XtX, Xty)

        a, b  = float(beta[0]), float(beta[1])
        A     = math.exp(a)
        alpha = -b                         # positive exponent = negative slope

        # Keep only physically meaningful (positive, finite) parameters
        if not (math.isfinite(A) and A > 0 and math.isfinite(alpha) and alpha > 0):
            continue

        # Raw-loss RMSE over observed points
        L_pred = E + A * D.pow(-alpha)
        rmse   = float(torch.sqrt(((L - L_pred) ** 2).mean()))

        # Strict less-than preserves first-seen order on exact ties
        if rmse < best_rmse:
            best_rmse = rmse
            best      = (E, A, alpha)

    E, A, alpha = best
    predictions = (E + A * T.pow(-alpha))   # float64 tensor

    return {
        "floor":       float(E),
        "coefficient": float(A),
        "exponent":    float(alpha),
        "predictions": predictions,
        "rmse":        float(best_rmse),
    }