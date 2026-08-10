def pad_and_center_crop(image, pad, crop_h, crop_w):
    """
    Returns: 2D list of lists of floats with shape (crop_h, crop_w), each rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0

    padded_h = H + 2 * pad
    padded_w = W + 2 * pad

    # Zero-pad
    padded = [[0.0] * padded_w for _ in range(padded_h)]
    for i in range(H):
        for j in range(W):
            padded[i + pad][j + pad] = float(image[i][j])

    # Center-crop start indices via floor division
    r_start = (padded_h - crop_h) // 2
    c_start = (padded_w - crop_w) // 2

    result = []
    for i in range(r_start, r_start + crop_h):
        row = [round(padded[i][j], 4) for j in range(c_start, c_start + crop_w)]
        result.append(row)

    return result