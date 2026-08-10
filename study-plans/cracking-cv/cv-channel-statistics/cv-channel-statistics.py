def channel_statistics(batch):
    """
    Returns: dict with keys "mean" and "std", each a list of length C, with every entry rounded to 4 decimals.
    """
    N = len(batch)
    H = len(batch[0]) if N > 0 else 0
    W = len(batch[0][0]) if H > 0 else 0
    C = len(batch[0][0][0]) if W > 0 else 0

    count = N * H * W

    sums = [0.0] * C
    for n in range(N):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    sums[c] += batch[n][h][w][c]

    means = [s / count for s in sums]

    sq_diffs = [0.0] * C
    for n in range(N):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    diff = batch[n][h][w][c] - means[c]
                    sq_diffs[c] += diff * diff

    variances = [sd / count for sd in sq_diffs]
    stds = [v ** 0.5 for v in variances]

    return {
        "mean": [round(m, 4) for m in means],
        "std": [round(s, 4) for s in stds],
    }