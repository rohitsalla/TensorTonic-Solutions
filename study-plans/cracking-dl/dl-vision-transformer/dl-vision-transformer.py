import numpy as np

def vit_patch_embed(image, patch_size, W_proj, b_proj, W_cls, W_pos):
    """
    Returns: Dict with "patches", "patch_embeddings", "tokens", "output", all rounded to 4 decimals.
    """
    image = np.array(image, dtype=float)
    W_proj = np.array(W_proj, dtype=float)
    b_proj = np.array(b_proj, dtype=float)
    W_cls = np.array(W_cls, dtype=float)
    W_pos = np.array(W_pos, dtype=float)

    C, H, W = image.shape
    ps = patch_size
    n_rows = H // ps
    n_cols = W // ps

    # Step 1: extract non-overlapping patches, row-first (top to bottom, left to right)
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch = image[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps]
            patches.append(patch.flatten())  # flattens (C, ps, ps) in C-order: channel, row, col
    patches = np.array(patches)  # (N, C*ps*ps)

    # Step 2: linear projection
    patch_embeddings = patches @ W_proj.T + b_proj

    # Step 3: prepend CLS token
    tokens = np.concatenate([W_cls, patch_embeddings], axis=0)

    # Step 4: add positional embedding
    output = tokens + W_pos

    return {
        "patches": np.round(patches, 4).tolist(),
        "patch_embeddings": np.round(patch_embeddings, 4).tolist(),
        "tokens": np.round(tokens, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }