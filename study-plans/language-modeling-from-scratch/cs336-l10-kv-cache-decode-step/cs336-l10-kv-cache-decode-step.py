import torch

def kv_cache_decode_step(query, new_key, new_value, key_cache, value_cache,
                         num_query_heads, num_kv_heads):
    """
    Returns: dictionary containing output and updated caches
    """
    B, Hq, D = query.shape
    Hkv      = num_kv_heads
    G        = Hq // Hkv       # query heads per kv head

    # ── Update caches: append new key/value at position S ────────────────
    # new_key/value: (B, Hkv, D) -> unsqueeze to (B, Hkv, 1, D)
    new_k = new_key.unsqueeze(2)    # (B, Hkv, 1, D)
    new_v = new_value.unsqueeze(2)  # (B, Hkv, 1, D)

    if key_cache.shape[2] == 0:
        new_key_cache   = new_k.clone()
        new_value_cache = new_v.clone()
    else:
        new_key_cache   = torch.cat([key_cache, new_k],   dim=2)  # (B, Hkv, S+1, D)
        new_value_cache = torch.cat([value_cache, new_v], dim=2)

    S_new = new_key_cache.shape[2]   # S+1 after append

    # ── Grouped query attention in float32 ───────────────────────────────
    q32 = query.float().reshape(B, Hkv, G, D)           # (B, Hkv, G, D)
    k32 = new_key_cache.float()                          # (B, Hkv, S+1, D)
    v32 = new_value_cache.float()                        # (B, Hkv, S+1, D)

    # Scores: q[b,h,g,:] · k[b,h,s,:] / sqrt(D)
    # einsum: (B, Hkv, G, D) x (B, Hkv, S+1, D) -> (B, Hkv, G, S+1)
    scores = torch.einsum("bhgd,bhsd->bhgs", q32, k32) * (D ** -0.5)
    attn   = torch.softmax(scores, dim=-1)               # (B, Hkv, G, S+1)

    # Weighted sum over sequence positions
    # einsum: (B, Hkv, G, S+1) x (B, Hkv, S+1, D) -> (B, Hkv, G, D)
    out = torch.einsum("bhgs,bhsd->bhgd", attn, v32)    # (B, Hkv, G, D)

    # Merge group axis back into query heads: (B, Hq, D)
    out = out.reshape(B, Hq, D).to(dtype=query.dtype)

    return {
        "output":          out,
        "new_key_cache":   new_key_cache.to(dtype=query.dtype),
        "new_value_cache": new_value_cache.to(dtype=query.dtype),
    }