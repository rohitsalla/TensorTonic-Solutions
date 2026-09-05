import torch
import math

def data_mixture_extrapolation(mixture_names, data_sizes, losses,
                               target_size, floor_candidates):
    """
    Returns: dictionary containing all fits and the winning mixture
    """
    def fit_single(D, L, floors):
        """Fit one mixture row; returns (floor, A, alpha, rmse)."""
        D = D.double()
        L = L.double()
        N = len(D)

        log_D = torch.log(D)
        ones  = torch.ones(N, dtype=torch.float64, device=D.device)
        X     = torch.stack([ones, log_D], dim=1)   # (N, 2)
        XtX   = X.T @ X                              # precompute once

        best_rmse = float('inf')
        best      = None

        for E_t in floors:
            E = float(E_t)
            if E >= float(L.min()):
                continue
            residuals = L - E
            if not (residuals > 0).all():
                continue

            log_r = torch.log(residuals)
            beta  = torch.linalg.solve(XtX, X.T @ log_r)
            a, b  = float(beta[0]), float(beta[1])
            A     = math.exp(a)
            alpha = -b

            if not (math.isfinite(A) and A > 0 and math.isfinite(alpha) and alpha > 0):
                continue

            L_pred = E + A * D.pow(-alpha)
            rmse   = float(torch.sqrt(((L - L_pred) ** 2).mean()))

            if rmse < best_rmse:
                best_rmse = rmse
                best      = (E, A, alpha)

        return best[0], best[1], best[2], best_rmse

    # ── Fit each mixture independently ───────────────────────────────────
    fits             = []
    target_pred_list = []
    T                = float(target_size)

    for k, name in enumerate(mixture_names):
        D = data_sizes[k]
        L = losses[k]

        E, A, alpha, rmse = fit_single(D, L, floor_candidates)

        pred = E + A * (T ** (-alpha))
        target_pred_list.append(pred)

        fits.append({
            "name":        name,
            "floor":       float(E),
            "coefficient": float(A),
            "exponent":    float(alpha),
            "rmse":        float(rmse),
        })

    # ── Build float64 predictions tensor ─────────────────────────────────
    target_predictions = torch.tensor(target_pred_list, dtype=torch.float64)

    # ── Winner: lowest prediction; mixture order breaks ties ─────────────
    winning_index   = int(torch.argmin(target_predictions).item())
    winning_mixture = mixture_names[winning_index]

    return {
        "fits":              fits,
        "target_predictions": target_predictions,
        "winning_mixture":   winning_mixture,
        "winning_index":     winning_index,
    }