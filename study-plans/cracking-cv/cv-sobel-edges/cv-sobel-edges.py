import numpy as np

def sobel_edges(image):
    """
    Returns: dict with keys "gx", "gy", "magnitude", each a list of lists of shape (H, W), with every entry rounded to 4 decimals
    """
    x = np.array(image, dtype=float)
    H, W = x.shape

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    x_padded = np.pad(x, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    gx = np.zeros((H, W))
    gy = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            window = x_padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(window * Gx)
            gy[i, j] = np.sum(window * Gy)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return {
        "gx": np.round(gx, 4).tolist(),
        "gy": np.round(gy, 4).tolist(),
        "magnitude": np.round(magnitude, 4).tolist(),
    }