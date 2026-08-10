import math

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """
    Returns: list of shape (grid_h * grid_w, embed_dim), 2D sin-cos positional embeddings, rounded to 4 decimals
    """
    half_dim = embed_dim // 2
    quarter_dim = half_dim // 2

    freqs = [1.0 / (10000 ** (2 * i / half_dim)) for i in range(quarter_dim)]

    def emb_1d(p):
        sin_part = [math.sin(p * w) for w in freqs]
        cos_part = [math.cos(p * w) for w in freqs]
        return sin_part + cos_part

    result = []
    for r in range(grid_h):
        row_emb = emb_1d(r)
        for c in range(grid_w):
            col_emb = emb_1d(c)
            vec = row_emb + col_emb
            result.append([round(float(v), 4) for v in vec])

    return result