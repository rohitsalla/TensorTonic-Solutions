import numpy as np

def adaboost_classify(X_train, y_train, X_test, n_estimators=10, seed=42):
    """
    Returns: list of predicted labels in {-1, +1} for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_test = np.array(X_test, dtype=float)

    n, d = X_train.shape
    weights = np.full(n, 1.0 / n)

    stumps = []  # list of (feature, threshold, polarity, alpha)

    for t in range(n_estimators):
        best_error = None
        best_feature = None
        best_threshold = None
        best_polarity = None

        for j in range(d):
            thresholds = np.unique(X_train[:, j])
            for th in thresholds:
                for polarity in [1, -1]:
                    # polarity=1: predict +1 if x > th else -1
                    # polarity=-1: predict +1 if x <= th else -1
                    if polarity == 1:
                        pred = np.where(X_train[:, j] > th, 1.0, -1.0)
                    else:
                        pred = np.where(X_train[:, j] <= th, 1.0, -1.0)

                    error = np.sum(weights[pred != y_train])

                    if best_error is None or error < best_error:
                        best_error = error
                        best_feature = j
                        best_threshold = th
                        best_polarity = polarity

        eps = best_error
        eps = min(max(eps, 1e-10), 1 - 1e-10)  # avoid log(0) / division by zero
        alpha = 0.5 * np.log((1 - eps) / eps)

        if best_polarity == 1:
            pred = np.where(X_train[:, best_feature] > best_threshold, 1.0, -1.0)
        else:
            pred = np.where(X_train[:, best_feature] <= best_threshold, 1.0, -1.0)

        weights = weights * np.exp(-alpha * y_train * pred)
        weights = weights / np.sum(weights)

        stumps.append((best_feature, best_threshold, best_polarity, alpha))

    predictions = []
    for x in X_test:
        total = 0.0
        for (feature, threshold, polarity, alpha) in stumps:
            if polarity == 1:
                h = 1.0 if x[feature] > threshold else -1.0
            else:
                h = 1.0 if x[feature] <= threshold else -1.0
            total += alpha * h
        predictions.append(1 if total >= 0 else -1)

    return predictions