def nms(boxes, scores, iou_threshold):
    """
    Returns: Python list of int indices of kept boxes, in score-descending pick order
    """
    n = len(boxes)

    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        union = area(b1) + area(b2) - inter
        return inter / union if union > 0 else 0.0

    # sort by score descending; ties -> lower original index first
    order = sorted(range(n), key=lambda i: (-scores[i], i))

    remaining = order[:]
    keep = []

    while remaining:
        current = remaining[0]
        keep.append(current)

        rest = remaining[1:]
        new_remaining = []
        for j in rest:
            if iou(boxes[current], boxes[j]) <= iou_threshold:
                new_remaining.append(j)
        remaining = new_remaining

    return [int(i) for i in keep]