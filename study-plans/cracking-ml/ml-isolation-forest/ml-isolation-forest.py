import numpy as np

def isolation_forest(
    X,
    n_estimators=100,
    max_samples=256,
    seed=42
):
    """
    Returns a list of anomaly scores rounded to 4 decimal places.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    sample_size = min(max_samples, n)
    max_depth = int(np.ceil(np.log2(sample_size)))
    rng = np.random.RandomState(seed)

    def c(size):
        if size <= 1:
            return 0.0
        if size == 2:
            return 1.0

        return (
            2.0 * (
                np.log(size - 1)
                + 0.5772156649
            )
            - 2.0 * (size - 1) / size
        )

    def build_tree(indices, depth):
        node_size = len(indices)

        if depth >= max_depth or node_size <= 1:
            return ("leaf", node_size)

        feature = rng.randint(d)
        values = X[indices, feature]

        minimum = np.min(values)
        maximum = np.max(values)

        # The selected feature cannot split this node.
        if minimum == maximum:
            return ("leaf", node_size)

        threshold = rng.uniform(minimum, maximum)

        left_mask = values < threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        return (
            "node",
            feature,
            threshold,
            build_tree(left_indices, depth + 1),
            build_tree(right_indices, depth + 1)
        )

    def path_length(point, node, depth=0):
        if node[0] == "leaf":
            node_size = node[1]
            return depth + c(node_size)

        _, feature, threshold, left, right = node

        if point[feature] < threshold:
            return path_length(point, left, depth + 1)

        return path_length(point, right, depth + 1)

    total_path_lengths = np.zeros(n, dtype=float)

    for _ in range(n_estimators):
        # Avoid consuming random numbers when every point is used.
        # This is required to match the seeded reference outputs.
        if sample_size == n:
            indices = np.arange(n)
        else:
            indices = rng.choice(
                n,
                size=sample_size,
                replace=False
            )

        tree = build_tree(indices, depth=0)

        for i in range(n):
            total_path_lengths[i] += path_length(X[i], tree)

    average_paths = total_path_lengths / n_estimators
    normalization = c(sample_size)

    if normalization == 0:
        scores = np.ones(n)
    else:
        scores = 2.0 ** (-average_paths / normalization)

    return np.round(scores, 4).tolist()