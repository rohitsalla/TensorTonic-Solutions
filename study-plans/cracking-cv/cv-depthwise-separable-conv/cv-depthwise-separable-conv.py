def depthwise_separable_conv(image, depthwise_weight, pointwise_weight, bias, stride, padding):
    """
    Returns: 3D list of shape (C_out, H_dw, W_dw), depthwise separable conv output rounded to 4 decimals
    """
    C_in = len(image)
    H = len(image[0])
    W = len(image[0][0])

    kH = len(depthwise_weight[0][0])
    kW = len(depthwise_weight[0][0][0])

    s = stride
    p = padding

    # Zero-pad each channel
    padded_h = H + 2 * p
    padded_w = W + 2 * p
    padded = [[[0.0] * padded_w for _ in range(padded_h)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H):
            for j in range(W):
                padded[c][i + p][j + p] = float(image[c][i][j])

    H_dw = (H + 2 * p - kH) // s + 1
    W_dw = (W + 2 * p - kW) // s + 1

    # Step 1: depthwise convolution, per channel independently
    dw = [[[0.0] * W_dw for _ in range(H_dw)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H_dw):
            for j in range(W_dw):
                row_start = i * s
                col_start = j * s
                total = 0.0
                for a in range(kH):
                    for b in range(kW):
                        total += depthwise_weight[c][0][a][b] * padded[c][row_start + a][col_start + b]
                dw[c][i][j] = total

    # Step 2: pointwise (1x1) convolution across channels
    C_out = len(pointwise_weight)
    result = []
    for co in range(C_out):
        channel_out = []
        for i in range(H_dw):
            row = []
            for j in range(W_dw):
                total = bias[co]
                for ci in range(C_in):
                    total += pointwise_weight[co][ci][0][0] * dw[ci][i][j]
                row.append(round(float(total), 4))
            channel_out.append(row)
        result.append(channel_out)

    return result