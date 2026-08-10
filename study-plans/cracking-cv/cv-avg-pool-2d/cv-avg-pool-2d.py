def avg_pool_2d(image, kernel_size, stride):
    """
    Returns: 2D list of shape (H_out, W_out), average-pooled values rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    k = kernel_size
    s = stride

    H_out = (H - k) // s + 1
    W_out = (W - k) // s + 1

    result = []
    for i in range(H_out):
        row = []
        for j in range(W_out):
            row_start = i * s
            col_start = j * s
            total = 0.0
            for u in range(k):
                for v in range(k):
                    total += image[row_start + u][col_start + v]
            avg = total / (k * k)
            row.append(round(float(avg), 4))
        result.append(row)

    return result