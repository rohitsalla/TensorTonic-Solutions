def normalize_image(image, mean, std):
    """
    Returns: 3D list of shape (H, W, C), each value rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    C = len(mean)

    result = []
    for i in range(H):
        row = []
        for j in range(W):
            pixel = []
            for c in range(C):
                v = (image[i][j][c] - mean[c]) / std[c]
                pixel.append(round(v, 4))
            row.append(pixel)
        result.append(row)

    return result