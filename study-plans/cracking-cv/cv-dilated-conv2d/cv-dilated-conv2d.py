def dilated_conv2d(image, weight, bias, dilation):
    """
    Returns: 3D Python list of shape (C_out, H_out, W_out), each value rounded to 4 decimals
    """
    C_in = len(image)
    H = len(image[0])
    W = len(image[0][0])

    C_out = len(weight)
    kH = len(weight[0][0])
    kW = len(weight[0][0][0])

    d = dilation

    H_out = H - (kH - 1) * d
    W_out = W - (kW - 1) * d

    result = []
    for co in range(C_out):
        channel_out = []
        for i in range(H_out):
            row = []
            for j in range(W_out):
                total = bias[co]
                for ci in range(C_in):
                    for p in range(kH):
                        for q in range(kW):
                            total += weight[co][ci][p][q] * image[ci][i + p * d][j + q * d]
                row.append(round(float(total), 4))
            channel_out.append(row)
        result.append(channel_out)

    return result