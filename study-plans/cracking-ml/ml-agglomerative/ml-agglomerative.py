import numpy as np

def agglomerative(X, n_clusters=2, linkage='single'):
    """
    Returns: list of integer cluster labels
    """
    X = np.asarray(X, dtype=float)
    n = len(X)

    clusters = {i: [i] for i in range(n)}

    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((X[i] - X[j])**2))
            dist[i, j] = d
            dist[j, i] = d

    active = set(range(n))
    next_id = n

    while len(active) > n_clusters:
        min_d = np.inf
        mi, mj = -1, -1
        active_list = sorted(active)
        for i_idx in range(len(active_list)):
            for j_idx in range(i_idx+1, len(active_list)):
                ci, cj = active_list[i_idx], active_list[j_idx]
                if dist[ci, cj] < min_d:
                    min_d = dist[ci, cj]
                    mi, mj = ci, cj

        new_members = clusters[mi] + clusters[mj]
        clusters[next_id] = new_members

        old_size = dist.shape[0]
        new_dist = np.full((next_id+1, next_id+1), np.inf)
        new_dist[:old_size, :old_size] = dist

        for ck in active:
            if ck == mi or ck == mj:
                continue
            if linkage == 'single':
                d = min(dist[mi, ck], dist[mj, ck])
            elif linkage == 'complete':
                d = max(dist[mi, ck], dist[mj, ck])
            elif linkage == 'average':
                d = (len(clusters[mi]) * dist[mi, ck] + len(clusters[mj]) * dist[mj, ck]) / (len(clusters[mi]) + len(clusters[mj]))
            new_dist[next_id, ck] = d
            new_dist[ck, next_id] = d

        dist = new_dist
        active.discard(mi)
        active.discard(mj)
        active.add(next_id)
        next_id += 1

    labels = np.zeros(n, dtype=int)
    for label, cluster_id in enumerate(sorted(active)):
        for idx in clusters[cluster_id]:
            labels[idx] = label

    return labels.tolist()
