def fpn_top_down_fusion(feature_maps):
    """
    Returns: list of L 3D Python lists [P[0], ..., P[L-1]], each value rounded to 4 decimals
    """
    L = len(feature_maps)

    def upsample2(feat):
        C = len(feat)
        H = len(feat[0])
        W = len(feat[0][0])
        out = [[[0.0] * (W * 2) for _ in range(H * 2)] for _ in range(C)]
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    val = feat[c][i][j]
                    out[c][2*i][2*j] = val
                    out[c][2*i][2*j+1] = val
                    out[c][2*i+1][2*j] = val
                    out[c][2*i+1][2*j+1] = val
        return out

    def add(a, b):
        C = len(a)
        H = len(a[0])
        W = len(a[0][0])
        return [[[a[c][i][j] + b[c][i][j] for j in range(W)] for i in range(H)] for c in range(C)]

    P = [None] * L
    P[L - 1] = feature_maps[L - 1]

    for l in range(L - 2, -1, -1):
        up = upsample2(P[l + 1])
        P[l] = add(feature_maps[l], up)

    result = []
    for p in P:
        C = len(p)
        H = len(p[0])
        W = len(p[0][0])
        rounded = [[[round(float(p[c][i][j]), 4) for j in range(W)] for i in range(H)] for c in range(C)]
        result.append(rounded)

    return result