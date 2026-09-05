import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    B, S_q, D = q.shape
    S_k = k.shape[1]
    d_h = D // num_heads

    # Split model width into heads, move head axis before sequence axis
    # (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)
    q_ = q.reshape(B, S_q, num_heads, d_h).transpose(1, 2)
    k_ = k.reshape(B, S_k, num_heads, d_h).transpose(1, 2)

    # Contract over head-width dim r; keep query (i) and key (j) axes separate
    # Result shape: (B, H, S_q, S_k)
    scores = torch.einsum("bhir,bhjr->bhij", q_, k_) / (d_h ** 0.5)

    return scores.to(dtype=q.dtype)