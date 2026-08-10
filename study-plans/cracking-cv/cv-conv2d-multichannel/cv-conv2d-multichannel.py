def conv2d(image, weight, bias, stride, padding):
    """
    Returns: 3D list of shape (Cout, H_out, W_out), each value rounded to 4 decimals
    """
    C_in = len(image)
    H = len(image[0])
    W = len(image[0][0])

    C_out = len(weight)
    kH = len(weight[0][0])
    kW = len(weight[0][0][0])

    s = stride
    P = padding

    # Zero-pad the image
    padded_h = H + 2 * P
    padded_w = W + 2 * P
    padded = [[[0.0] * padded_w for _ in range(padded_h)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H):
            for j in range(W):
                padded[c][i + P][j + P] = float(image[c][i][j])

    H_out = (H + 2 * P - kH) // s + 1
    W_out = (W + 2 * P - kW) // s + 1

    result = []
    for co in range(C_out):
        channel_out = []
        for i in range(H_out):
            row = []
            for j in range(W_out):
                total = bias[co]
                row_start = i * s
                col_start = j * s
                for ci in range(C_in):
                    for p in range(kH):
                        for q in range(kW):
                            total += weight[co][ci][p][q] * padded[ci][row_start + p][col_start + q]
                row.append(round(float(total), 4))
            channel_out.append(row)
        result.append(channel_out)

    return result