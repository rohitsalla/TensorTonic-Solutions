import numpy as np

def encode_and_decode_boxes(proposals, gt_boxes):
    """
    Returns: dict with keys "deltas" (N x 4) and "decoded" (N x 4 xyxy), each rounded to 4 decimals
    """
    proposals = np.array(proposals, dtype=float)
    gt_boxes = np.array(gt_boxes, dtype=float)

    def to_center_size(boxes):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h

    cx_p, cy_p, w_p, h_p = to_center_size(proposals)
    cx_g, cy_g, w_g, h_g = to_center_size(gt_boxes)

    # Encode
    tx = (cx_g - cx_p) / w_p
    ty = (cy_g - cy_p) / h_p
    tw = np.log(w_g / w_p)
    th = np.log(h_g / h_p)

    deltas = np.stack([tx, ty, tw, th], axis=1)

    # Decode (applying deltas back onto the same proposals)
    cx_d = cx_p + tx * w_p
    cy_d = cy_p + ty * h_p
    w_d = w_p * np.exp(tw)
    h_d = h_p * np.exp(th)

    x1_d = cx_d - w_d / 2
    y1_d = cy_d - h_d / 2
    x2_d = cx_d + w_d / 2
    y2_d = cy_d + h_d / 2

    decoded = np.stack([x1_d, y1_d, x2_d, y2_d], axis=1)

    return {
        "deltas": np.round(deltas, 4).tolist(),
        "decoded": np.round(decoded, 4).tolist(),
    }