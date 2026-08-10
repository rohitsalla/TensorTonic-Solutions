import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, dtype=float)

    classes = np.unique(y_train)
    n = X_train.shape[0]
    eps = 1e-9

    priors = {}
    means = {}
    variances = {}

    for c in classes:
        Xc = X_train[y_train == c]
        priors[c] = Xc.shape[0] / n
        means[c] = Xc.mean(axis=0)
        variances[c] = Xc.var(axis=0) + eps  # population variance (ddof=0)

    predictions = []
    for x in X_test:
        log_posteriors = {}
        for c in classes:
            mu = means[c]
            var = variances[c]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mu) ** 2) / var)
            log_posteriors[c] = np.log(priors[c]) + log_likelihood
        best_c = max(log_posteriors, key=log_posteriors.get)
        predictions.append(int(best_c))

    return predictions