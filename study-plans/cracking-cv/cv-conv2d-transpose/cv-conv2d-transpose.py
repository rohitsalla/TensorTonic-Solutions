def conv_transpose_2d(image, weight, bias, stride, padding):
    """
    Returns: 3D list of shape (C_out, H_out, W_out) holding the transposed convolution result, rounded to 4 decimals
    """
    C_in = len(image)
    H = len(image[0])
    W = len(image[0][0])

    C_out = len(weight[0])
    kH = len(weight[0][0])
    kW = len(weight[0][0][0])

    s = stride
    p = padding

    buf_h = (H - 1) * s + kH
    buf_w = (W - 1) * s + kW

    buf = [[[0.0] * buf_w for _ in range(buf_h)] for _ in range(C_out)]

    for c in range(C_in):
        for h in range(H):
            for w in range(W):
                val = image[c][h][w]
                for cout in range(C_out):
                    wt = weight[c][cout]
                    for pp in range(kH):
                        for qq in range(kW):
                            buf[cout][h * s + pp][w * s + qq] += wt[pp][qq] * val

    H_out = (H - 1) * s - 2 * p + kH
    W_out = (W - 1) * s - 2 * p + kW

    result = []
    for cout in range(C_out):
        channel_out = []
        for i in range(H_out):
            row = []
            for j in range(W_out):
                val = buf[cout][i + p][j + p] + bias[cout]
                row.append(round(float(val), 4))
            channel_out.append(row)
        result.append(channel_out)

    return result