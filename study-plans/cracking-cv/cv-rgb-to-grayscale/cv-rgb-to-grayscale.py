def rgb_to_grayscale(image):
    """
    Returns: 2D list of shape (H, W) with luma values rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0

    result = []
    for i in range(H):
        row = []
        for j in range(W):
            r, g, b = image[i][j]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            row.append(round(y, 4))
        result.append(row)

    return result