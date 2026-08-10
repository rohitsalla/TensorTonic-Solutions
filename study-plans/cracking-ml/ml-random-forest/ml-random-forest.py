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
    return int(candidates.min())

def _get_m(max_features, d):
    if max_features == 'sqrt':
        m = int(np.floor(np.sqrt(d)))
    elif max_features == 'log2':
        m = int(np.floor(np.log2(d)))
    elif isinstance(max_features, int):
        m = min(max_features, d)
    else:
        m = d
    return max(1, m)

def _build_tree(X, y, depth, max_depth, min_samples, max_features, rng):
    n = len(y)
    d = X.shape[1]

    if depth >= max_depth or n < min_samples or len(np.unique(y)) == 1:
        return {"leaf": True, "class": _majority_class(y)}

    m = _get_m(max_features, d)
    feature_subset = rng.choice(d, size=m, replace=False)

    parent_gini = _gini(y)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for j in feature_subset:
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

    left_subtree = _build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples, max_features, rng)
    right_subtree = _build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples, max_features, rng)

    return {
        "leaf": False,
        "feature": int(best_feature),
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

def random_forest_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, max_features='sqrt', seed=42):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, dtype=float)

    n = X_train.shape[0]
    rng = np.random.RandomState(seed)

    trees = []
    for _ in range(n_estimators):
        idx = rng.randint(0, n, size=n)
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        tree = _build_tree(X_boot, y_boot, 0, max_depth, 2, max_features, rng)
        trees.append(tree)

    predictions = []
    for x in X_test:
        votes = np.array([_predict_one(tree, x) for tree in trees])
        classes, counts = np.unique(votes, return_counts=True)
        max_count = counts.max()
        winners = classes[counts == max_count]
        predictions.append(int(winners.min()))

    return predictions