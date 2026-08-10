import numpy as np

def vit_attention(x, W_qkv, b_qkv, W_o, b_o, num_heads):
    """
    Returns: 3D Python list of shape (B, N, D), each entry rounded to 4 decimals
    """
    x = np.array(x, dtype=float)          # (B,N,D)
    W_qkv = np.array(W_qkv, dtype=float)  # (D,3D)
    b_qkv = np.array(b_qkv, dtype=float)  # (3D,)
    W_o = np.array(W_o, dtype=float)      # (D,D)
    b_o = np.array(b_o, dtype=float)      # (D,)

    B, N, D = x.shape
    h = num_heads
    dh = D // h

    # Combined QKV projection, then column-split into Q, K, V
    qkv = x @ W_qkv + b_qkv  # (B,N,3D)
    Q = qkv[:, :, :D]
    K = qkv[:, :, D:2*D]
    V = qkv[:, :, 2*D:3*D]

    def split_heads(t):
        # (B,N,D) -> (B,N,h,dh) -> (B,h,N,dh)
        return t.reshape(B, N, h, dh).transpose(0, 2, 1, 3)

    Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

    # Scaled dot-product attention per head
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(dh)  # (B,h,N,N)
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = weights @ Vh  # (B,h,N,dh)

    # Merge heads back and project through W_o
    context = context.transpose(0, 2, 1, 3).reshape(B, N, D)  # (B,N,D)
    out = context @ W_o + b_o

    return np.round(out, 4).tolist()