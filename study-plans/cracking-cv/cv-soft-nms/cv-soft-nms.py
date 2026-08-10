import math

def soft_nms(boxes, scores, sigma, score_threshold):
    """
    Returns: list of [int_index, float_decayed_score] pairs in pick order, scores rounded to 4 decimals at pick time
    """
    n = len(boxes)
    scores = list(scores)  # unrounded, mutable working copy
    alive = list(range(n))

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

    output = []
    while alive:
        # Pick candidate with highest current score; ties -> lowest index
        alive.sort(key=lambda i: (-scores[i], i))
        current = alive[0]

        if scores[current] <= score_threshold:
            break

        output.append([int(current), round(float(scores[current]), 4)])

        alive = alive[1:]
        for j in alive:
            v = iou(boxes[current], boxes[j])
            scores[j] = scores[j] * math.exp(-(v ** 2) / sigma)

        # Drop any candidate that has decayed to/below the threshold
        alive = [j for j in alive if scores[j] > score_threshold]

    return output