import math

def cosine_topk(queries, corpus, k):
    """
    Returns: dict with keys 'indices' (M x k ints) and 'scores' (M x k floats rounded to 4 decimals)
    """
    eps = 1e-12

    def norm(vec):
        return math.sqrt(sum(v * v for v in vec))

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(len(a)))

    q_norms = [norm(q) for q in queries]
    c_norms = [norm(c) for c in corpus]

    indices_out = []
    scores_out = []

    for i, q in enumerate(queries):
        sims = []
        for j, c in enumerate(corpus):
            s = dot(q, c) / ((q_norms[i] + eps) * (c_norms[j] + eps))
            sims.append((j, s))

        # sort by descending score, tie-break by lower index
        sims.sort(key=lambda x: (-x[1], x[0]))

        top = sims[:k]
        indices_out.append([int(idx) for idx, _ in top])
        scores_out.append([round(float(s), 4) for _, s in top])

    return {
        "indices": indices_out,
        "scores": scores_out,
    }