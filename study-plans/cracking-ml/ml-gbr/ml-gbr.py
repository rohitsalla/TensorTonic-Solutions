import numpy as np

def _mse(y):
    if len(y) == 0:
        return 0.0
    return np.mean((y - np.mean(y)) ** 2)

def _build_tree(X, y, depth, max_depth, min_samples):
    n = len(y)

    if depth >= max_depth or n < min_samples or len(np.unique(y)) == 1:
        return {"leaf": True, "value": float(np.mean(y))}

    parent_mse = _mse(y)
    best_reduction = 0.0
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]
    for j in range(n_features):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            left_mask = X[:, j] <= t
            right_mask = ~left_mask

            n_left = left_mask.sum()
            n_right = right_mask.sum()

            if n_left == 0 or n_right == 0:
                continue

            mse_left = _mse(y[left_mask])
            mse_right = _mse(y[right_mask])

            weighted_mse = (n_left / n) * mse_left + (n_right / n) * mse_right
            reduction = parent_mse - weighted_mse

            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = j
                best_threshold = t

    if best_feature is None or best_reduction <= 0:
        return {"leaf": True, "value": float(np.mean(y))}

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_subtree = _build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples)
    right_subtree = _build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples)

    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_subtree,
        "right": right_subtree,
    }

def _predict_one(node, x):
    while not node["leaf"]:
        if x[node["feature"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["value"]

def _predict_tree(tree, X):
    return np.array([_predict_one(tree, x) for x in X])

def gbr_predict(X_train, y_train, X_test, n_estimators=10, max_depth=3, learning_rate=0.1, seed=42):
    """
    Returns: list of predicted values rounded to 4 decimal places
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_test = np.array(X_test, dtype=float)

    F0 = np.mean(y_train)
    F_train = np.full(y_train.shape, F0)
    F_test = np.full(X_test.shape[0], F0)

    for _ in range(n_estimators):
        residuals = y_train - F_train
        tree = _build_tree(X_train, residuals, 0, max_depth, min_samples=2)

        train_pred = _predict_tree(tree, X_train)
        test_pred = _predict_tree(tree, X_test)

        F_train = F_train + learning_rate * train_pred
        F_test = F_test + learning_rate * test_pred

    return [round(float(v), 4) for v in F_test]