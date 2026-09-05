import torch
import math

def hyperparameter_scaling_fit(model_scales, learning_rates, batch_sizes,
                               measured_losses, target_scale):
    """
    Returns: dictionary containing measured optima and two scaling laws
    """
    S  = model_scales.double()
    LR = learning_rates.double()
    BS = batch_sizes.double()
    L  = measured_losses.double()   # (n_scales, n_lr, n_bs)

    n_scales, n_lr, n_bs = L.shape

    # ── Select per-scale optimum: argmin in LR-major (row-major) order ───
    optima  = []
    opt_lr  = []
    opt_bs  = []

    for i in range(n_scales):
        grid     = L[i]                            # (n_lr, n_bs)
        flat_idx = int(grid.reshape(-1).argmin())   # LR-major = row-major
        lr_idx, bs_idx = divmod(flat_idx, n_bs)

        optima.append({
            "model_scale":   float(S[i]),
            "learning_rate": float(LR[lr_idx]),
            "batch_size":    float(BS[bs_idx]),
            "loss":          float(grid[lr_idx, bs_idx]),
        })
        opt_lr.append(float(LR[lr_idx]))
        opt_bs.append(float(BS[bs_idx]))

    # ── Log-log OLS helper ────────────────────────────────────────────────
    def log_log_fit(x_vals, y_vals):
        log_x = torch.log(torch.tensor(x_vals, dtype=torch.float64))
        log_y = torch.log(torch.tensor(y_vals, dtype=torch.float64))
        n     = len(log_x)
        ones  = torch.ones(n, dtype=torch.float64)
        X     = torch.stack([ones, log_x], dim=1)   # (n, 2)
        beta  = torch.linalg.solve(X.T @ X, X.T @ log_y)
        log_a, b = float(beta[0]), float(beta[1])
        a         = math.exp(log_a)
        residuals = log_y - (log_a + b * log_x)     # log-space residuals
        return a, b, residuals

    # ── Fit both hyperparameters ──────────────────────────────────────────
    s_vals = [float(s) for s in S]

    a_lr, b_lr, res_lr = log_log_fit(s_vals, opt_lr)
    a_bs, b_bs, res_bs = log_log_fit(s_vals, opt_bs)

    # ── Target predictions ────────────────────────────────────────────────
    T = float(target_scale)
    target_lr = a_lr * (T ** b_lr)
    target_bs = a_bs * (T ** b_bs)

    return {
        "optima":                     optima,
        "learning_rate_coefficient":  float(a_lr),
        "learning_rate_exponent":     float(b_lr),
        "batch_size_coefficient":     float(a_bs),
        "batch_size_exponent":        float(b_bs),
        "learning_rate_residuals":    res_lr,
        "batch_size_residuals":       res_bs,
        "target_learning_rate":       float(target_lr),
        "target_batch_size":          float(target_bs),
    }