import numpy as np

def permutation_importance(X, y, predict_fn, n_repeats=5, seed=42):
    """
    Returns: list of importance scores (one per feature) rounded to 4 decimal places
    """
    X = np.array(X, dtype=float)
    y = np.array(y)
    n, d = X.shape

    rng = np.random.RandomState(seed)

    baseline_pred = np.array(predict_fn(X))
    baseline_acc = np.mean(baseline_pred == y)

    importances = []
    for j in range(d):
        drops = []
        for r in range(n_repeats):
            X_permuted = X.copy()
            perm = rng.permutation(n)
            X_permuted[:, j] = X_permuted[perm, j]

            pred = np.array(predict_fn(X_permuted))
            acc = np.mean(pred == y)

            drops.append(baseline_acc - acc)

        importances.append(round(float(np.mean(drops)), 4))

    return importances