def mixup(image_a, image_b, label_a, label_b, lam):
    """
    Returns: dict with keys "image" (C,H,W list) and "label" (length-K list), values rounded to 4 decimals
    """
    C = len(image_a)
    H = len(image_a[0])
    W = len(image_a[0][0])

    mixed_image = [
        [
            [round(lam * image_a[c][h][w] + (1 - lam) * image_b[c][h][w], 4) for w in range(W)]
            for h in range(H)
        ]
        for c in range(C)
    ]

    mixed_label = [round(lam * label_a[k] + (1 - lam) * label_b[k], 4) for k in range(len(label_a))]

    return {
        "image": mixed_image,
        "label": mixed_label,
    }