import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    """
    Returns: list of integer labels (-1 for noise)
    """
    X = np.array(X, dtype=float)
    n = X.shape[0]

    labels = np.full(n, -2, dtype=int)  # -2 = unvisited, -1 = noise, >=0 = cluster id

    def region_query(idx):
        dists = np.sqrt(np.sum((X - X[idx]) ** 2, axis=1))
        return list(np.where(dists <= eps)[0])

    cluster_id = 0

    for i in range(n):
        if labels[i] != -2:
            continue  # already processed

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1  # tentatively noise (may become a border point later)
            continue

        # i is a core point: start a new cluster and expand it
        labels[i] = cluster_id
        seeds = [pt for pt in neighbors if pt != i]

        k = 0
        while k < len(seeds):
            q = seeds[k]
            if labels[q] == -1:
                labels[q] = cluster_id  # was noise, now a border point of this cluster
            elif labels[q] == -2:
                labels[q] = cluster_id
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    # q is also a core point: add its neighbors to the expansion frontier
                    for pt in q_neighbors:
                        if pt not in seeds:
                            seeds.append(pt)
            k += 1

        cluster_id += 1

    return labels.tolist()