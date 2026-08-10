def dice_loss(pred, target, smooth):
    """
    Returns: float, the Dice loss 1 - dice, rounded to 4 decimals
    """
    H = len(pred)
    W = len(pred[0]) if H > 0 else 0

    intersection = 0.0
    pred_sum = 0.0
    target_sum = 0.0

    for i in range(H):
        for j in range(W):
            p = pred[i][j]
            t = target[i][j]
            intersection += p * t
            pred_sum += p
            target_sum += t

    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    loss = 1 - dice

    return round(float(loss), 4)