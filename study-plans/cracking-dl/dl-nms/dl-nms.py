import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Returns: Dict with "kept_indices" and "kept_scores", values rounded to 4 decimals.
    """
    if len(boxes) == 0:
        return {"kept_indices": [], "kept_scores": []}

    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    n = len(boxes)

    def iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        return intersection / (union + 1e-8)

    # sort indices by descending score
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    remaining = order[:]
    kept_indices = []
    kept_scores = []

    while remaining:
        current = remaining[0]
        kept_indices.append(current)
        kept_scores.append(scores[current])

        rest = remaining[1:]
        new_remaining = []
        for idx in rest:
            if iou(boxes[current], boxes[idx]) <= iou_threshold:
                new_remaining.append(idx)
        remaining = new_remaining

    return {
        "kept_indices": [int(i) for i in kept_indices],
        "kept_scores": [round(float(s), 4) for s in kept_scores],
    }