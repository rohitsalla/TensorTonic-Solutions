import torch

def causal_gqa(q, k, v):
    """
    Returns: causal grouped-query attention tensor of shape (B, H_q, S, D)
    """
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    G = H_q // H_kv                          # queries per key/value head

    # Work in float32 throughout
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()

    # Reshape q: (B, H_kv, G, S, D) — groups contiguous query heads together
    q32 = q32.reshape(B, H_kv, G, S, D)

    # Grouped scores: (B, H_kv, G, S_q, S_k)
    # Each query group contracts with its shared key head over D
    scale  = D ** -0.5
    scores = torch.einsum("bkgid,bkjd->bkgij", q32, k32) * scale

    # Causal mask: block positions where key index j > query index i
    mask = torch.ones(S, S, dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))

    # Softmax over key positions, then weighted sum over values
    probs = torch.softmax(scores, dim=-1)                  # (B, H_kv, G, S, S)
    out   = torch.einsum("bkgij,bkjd->bkgid", probs, v32) # (B, H_kv, G, S, D)

    # Merge group axis back into query heads: (B, H_q, S, D)
    return out.reshape(B, H_q, S, D).to(dtype=q.dtype)