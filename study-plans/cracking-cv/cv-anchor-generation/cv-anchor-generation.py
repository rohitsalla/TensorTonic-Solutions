import math

def generate_anchors(feature_h, feature_w, stride, base_size, scales, aspect_ratios):
    """
    Returns: 2D Python list of shape (feature_h*feature_w*len(scales)*len(aspect_ratios), 4) of xyxy anchors rounded to 4 decimals
    """
    anchors = []
    b = base_size
    s = stride

    for fy in range(feature_h):
        for fx in range(feature_w):
            cx = (fx + 0.5) * s
            cy = (fy + 0.5) * s
            for scale in scales:
                for r in aspect_ratios:
                    sqrt_r = math.sqrt(r)
                    w = b * scale * sqrt_r
                    h = b * scale / sqrt_r

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([
                        round(x1, 4),
                        round(y1, 4),
                        round(x2, 4),
                        round(y2, 4),
                    ])

    return anchors