import numpy as np

def _gini(y):
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def _majority_class(y):
    counts = np.bincount(y)
    max_count = counts.max()
    candidates = np.where(counts == max_count)[0]
    return int(candidates.min())  # ties broken by smallest label

def _build_tree(X, y, depth, max_depth, min_samples):
    n = len(y)

    # Stopping conditions
    if depth >= max_depth or n < min_samples or len(np.unique(y)) == 1:
        return {"leaf": True, "class": _majority_class(y)}

    parent_gini = _gini(y)
    best_gain = 0.0
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

            gini_left = _gini(y[left_mask])
            gini_right = _gini(y[right_mask])

            weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = t

    if best_feature is None or best_gain <= 0:
        return {"leaf": True, "class": _majority_class(y)}

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
    return node["class"]

def cart_classify(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, dtype=float)

    tree = _build_tree(X_train, y_train, 0, max_depth, min_samples)

    predictions = [_predict_one(tree, x) for x in X_test]
    return predictions