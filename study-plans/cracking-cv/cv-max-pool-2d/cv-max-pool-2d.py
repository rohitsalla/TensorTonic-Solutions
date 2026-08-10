def max_pool_2d(image, kernel_size, stride):
    """
    Returns: 2D list of shape (H_out, W_out), max-pooled values rounded to 4 decimals
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
            window_max = float('-inf')
            for a in range(k):
                for b in range(k):
                    val = image[row_start + a][col_start + b]
                    if val > window_max:
                        window_max = val
            row.append(round(float(window_max), 4))
        result.append(row)

    return result