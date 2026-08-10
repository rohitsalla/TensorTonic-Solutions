import numpy as np

def kmeans(X, k, max_iters=100, seed=42):
    """
    Returns: tuple of (labels as list[int], centroids as list[list[float]])
    """
    X = np.array(X, dtype=float)
    n, d = X.shape

    rng = np.random.RandomState(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = X[init_idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        distances = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        new_labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            mask = new_labels == j
            if np.any(mask):
                new_centroids[j] = X[mask].mean(axis=0)
            # else: keep centroid unchanged (no assigned points)

        labels = new_labels

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    labels_list = [int(v) for v in labels]
    centroids_list = [[round(float(v), 4) for v in row] for row in centroids]

    return (labels_list, centroids_list)
