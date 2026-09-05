import torch
import math

def isoflop_optimal_frontier(compute_budgets, parameter_counts, terminal_losses):
    """
    Returns: dictionary containing optima and the fitted frontier
    """
    C_arr = compute_budgets.double()
    n_budgets = len(C_arr)

    optima      = []
    opt_N_list  = []
    opt_C_list  = []

    for i in range(n_budgets):
        C = float(C_arr[i])
        N = parameter_counts[i].double()   # (M,)
        L = terminal_losses[i].double()    # (M,)
        M = len(N)

        x = torch.log(N)   # log-parameter axis

        # Fit quadratic: L ≈ a*x² + b*x + c via OLS
        X    = torch.stack([x**2, x, torch.ones(M, dtype=torch.float64, device=N.device)], dim=1)
        XtX  = X.T @ X                         # (3, 3)
        Xty  = X.T @ L                         # (3,)
        beta = torch.linalg.solve(XtX, Xty)   # [a, b, c]

        a, b, c = float(beta[0]), float(beta[1]), float(beta[2])

        # Vertex of parabola: x* = -b / (2a)
        x_star = -b / (2.0 * a)
        N_opt  = math.exp(x_star)
        D_opt  = C / (6.0 * N_opt)            # D = C / (6N) from C = 6ND
        L_opt  = a * x_star**2 + b * x_star + c

        optima.append({
            "compute":    float(C),
            "parameters": float(N_opt),
            "tokens":     float(D_opt),
            "loss":       float(L_opt),
        })
        opt_N_list.append(N_opt)
        opt_C_list.append(C)

    # ── Fit log-log frontier: log(N*) = log(k) + alpha * log(C) ──────────
    log_C  = torch.tensor([math.log(c) for c in opt_C_list], dtype=torch.float64)
    log_N  = torch.tensor([math.log(n) for n in opt_N_list], dtype=torch.float64)

    ones   = torch.ones(n_budgets, dtype=torch.float64)
    X2     = torch.stack([ones, log_C], dim=1)   # (K, 2): [intercept, slope]
    beta2  = torch.linalg.solve(X2.T @ X2, X2.T @ log_N)

    log_k, alpha = float(beta2[0]), float(beta2[1])
    k = math.exp(log_k)

    return {
        "optima":             optima,
        "frontier_scale":     float(k),
        "frontier_exponent":  float(alpha),
    }