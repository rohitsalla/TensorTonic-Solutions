import math

def gaussian_blur_2d(image, kernel_size, sigma):
    """
    Returns: 2D list of floats with shape (H, W), each entry rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    k = kernel_size
    center = (k - 1) / 2.0

    # 1D Gaussian
    g1d = [math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(k)]

    # 2D kernel as outer product, normalized
    G2d = [[g1d[i] * g1d[j] for j in range(k)] for i in range(k)]
    total = sum(sum(row) for row in G2d)
    G2d = [[v / total for v in row] for row in G2d]

    pad = (k - 1) // 2
    padded_h = H + 2 * pad
    padded_w = W + 2 * pad

    padded = [[0.0] * padded_w for _ in range(padded_h)]
    for i in range(H):
        for j in range(W):
            padded[i + pad][j + pad] = float(image[i][j])

    result = []
    for i in range(H):
        row = []
        for j in range(W):
            val = 0.0
            for ki in range(k):
                for kj in range(k):
                    val += padded[i + ki][j + kj] * G2d[ki][kj]
            row.append(round(val, 4))
        result.append(row)

    return result