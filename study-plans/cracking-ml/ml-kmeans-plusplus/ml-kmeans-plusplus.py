import numpy as np

def kmeans_plusplus(X, k, seed=42):
    """
    Returns: list of centroids, each a list of floats rounded to 4 decimal places
    """
    X = np.array(X, dtype=float)
    n, d = X.shape

    rng = np.random.RandomState(seed)

    first_idx = rng.randint(0, n)
    centroids = [X[first_idx]]

    for _ in range(1, k):
        centroids_arr = np.array(centroids)
        distances = np.sqrt(((X[:, None, :] - centroids_arr[None, :, :]) ** 2).sum(axis=2))
        min_dist = distances.min(axis=1)
        sq_dist = min_dist ** 2

        total = sq_dist.sum()
        probs = sq_dist / total

        next_idx = rng.choice(n, p=probs)
        centroids.append(X[next_idx])

    return [[round(float(v), 4) for v in c] for c in centroids]