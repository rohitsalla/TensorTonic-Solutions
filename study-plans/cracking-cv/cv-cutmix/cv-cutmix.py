def cutmix(image_a, image_b, bbox, label_a, label_b):
    """
    Returns: dict with keys 'image' (list shape (C,H,W)), 'label' (list length K), 'lam' (float), all numerics rounded to 4 decimals
    """
    C = len(image_a)
    H = len(image_a[0])
    W = len(image_a[0][0])

    x1, y1, x2, y2 = bbox

    mixed_image = [
        [
            [image_a[c][h][w] for w in range(W)]
            for h in range(H)
        ]
        for c in range(C)
    ]

    for c in range(C):
        for h in range(y1, y2):
            for w in range(x1, x2):
                mixed_image[c][h][w] = image_b[c][h][w]

    mixed_image = [
        [[round(float(mixed_image[c][h][w]), 4) for w in range(W)] for h in range(H)]
        for c in range(C)
    ]

    patch_area = (y2 - y1) * (x2 - x1)
    lam = 1 - patch_area / (H * W)

    mixed_label = [round(lam * label_a[k] + (1 - lam) * label_b[k], 4) for k in range(len(label_a))]

    return {
        "image": mixed_image,
        "label": mixed_label,
        "lam": round(float(lam), 4),
    }