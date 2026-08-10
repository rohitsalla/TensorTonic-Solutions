def mean_iou(pred, target, num_classes):
    """
    Returns: float, mean IoU over classes present in pred or target, rounded to 4 decimals (0.0 if none)
    """
    H = len(pred)
    W = len(pred[0]) if H > 0 else 0

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for i in range(H):
        for j in range(W):
            p = pred[i][j]
            t = target[i][j]
            if p == t:
                tp[p] += 1
            else:
                fp[p] += 1
                fn[t] += 1

    ious = []
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        if denom > 0:
            ious.append(tp[c] / denom)

    if not ious:
        return 0.0

    return round(float(sum(ious) / len(ious)), 4)