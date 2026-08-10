def patchify(images, patch_size):
    """
    Returns: 3D Python list of shape (B, num_patches, patch_size**2 * C), values rounded to 4 decimals
    """
    B = len(images)
    C = len(images[0])
    H = len(images[0][0])
    W = len(images[0][0][0])
    P = patch_size

    Nh = H // P
    Nw = W // P

    result = []
    for b in range(B):
        patches = []
        for h in range(Nh):
            for w in range(Nw):
                patch = []
                for p1 in range(P):
                    for p2 in range(P):
                        for c in range(C):
                            val = images[b][c][h * P + p1][w * P + p2]
                            patch.append(round(float(val), 4))
                patches.append(patch)
        result.append(patches)

    return result