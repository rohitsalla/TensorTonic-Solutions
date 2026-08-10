def bilinear_resize(image, new_h, new_w):
    """
    Returns: list of lists of floats, shape (new_h, new_w), each value rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0

    result = []
    for i in range(new_h):
        if new_h == 1:
            y_src = 0.0
        else:
            y_src = i * (H - 1) / (new_h - 1)

        y0 = int(y_src)
        y1 = min(y0 + 1, H - 1)
        wy = y_src - y0

        row = []
        for j in range(new_w):
            if new_w == 1:
                x_src = 0.0
            else:
                x_src = j * (W - 1) / (new_w - 1)

            x0 = int(x_src)
            x1 = min(x0 + 1, W - 1)
            wx = x_src - x0

            val = (
                (1 - wy) * (1 - wx) * image[y0][x0]
                + (1 - wy) * wx * image[y0][x1]
                + wy * (1 - wx) * image[y1][x0]
                + wy * wx * image[y1][x1]
            )
            row.append(round(val, 4))

        result.append(row)

    return result