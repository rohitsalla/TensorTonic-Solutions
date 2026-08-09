import numpy as np

def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma + beta

def _mha(Q_in, K_in, V_in, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """Multi-head attention with distinct Q/K/V input sources (supports self- or cross-attention)."""
    Q = Q_in @ W_q.T
    K = K_in @ W_k.T
    V = V_in @ W_v.T

    seq_q, d_model = Q.shape
    seq_k = K.shape[0]
    h = num_heads
    d_head = d_model // h

    def split_heads(t, seq):
        return t.reshape(seq, h, d_head).transpose(1, 0, 2)

    Qh, Kh, Vh = split_heads(Q, seq_q), split_heads(K, seq_k), split_heads(V, seq_k)

    mask_arr = np.array(mask, dtype=bool) if mask is not None else None

    head_outputs = []
    for hi in range(h):
        Qi, Ki, Vi = Qh[hi], Kh[hi], Vh[hi]
        scores = (Qi @ Ki.T) / np.sqrt(d_head)
        if mask_arr is not None:
            scores = np.where(mask_arr, scores, scores - 1e9)
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        head_outputs.append(weights @ Vi)

    concat = np.concatenate(head_outputs, axis=-1)
    return concat @ W_o.T

def decoder_block(x, enc_output, W_q1, W_k1, W_v1, W_o1, W_q2, W_k2, W_v2, W_o2, num_heads,
                   W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, gamma3, beta3, mask=None):
    """
    Implements one transformer decoder block.
    Returns: Dict with "self_attn", "norm1", "cross_attn", "norm2", "ffn_output", "output",
             all as list-of-lists rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    enc_output = np.array(enc_output, dtype=float)
    W_q1 = np.array(W_q1, dtype=float); W_k1 = np.array(W_k1, dtype=float)
    W_v1 = np.array(W_v1, dtype=float); W_o1 = np.array(W_o1, dtype=float)
    W_q2 = np.array(W_q2, dtype=float); W_k2 = np.array(W_k2, dtype=float)
    W_v2 = np.array(W_v2, dtype=float); W_o2 = np.array(W_o2, dtype=float)
    W1 = np.array(W1, dtype=float); b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float); b2 = np.array(b2, dtype=float)
    gamma1 = np.array(gamma1, dtype=float); beta1 = np.array(beta1, dtype=float)
    gamma2 = np.array(gamma2, dtype=float); beta2 = np.array(beta2, dtype=float)
    gamma3 = np.array(gamma3, dtype=float); beta3 = np.array(beta3, dtype=float)

    # --- Masked self-attention (Q = K = V = x) ---
    self_attn = _mha(x, x, x, W_q1, W_k1, W_v1, W_o1, num_heads, mask=mask)
    norm1 = _layer_norm(x + self_attn, gamma1, beta1)

    # --- Cross-attention (Q from decoder norm1, K/V from encoder output, no mask) ---
    cross_attn = _mha(norm1, enc_output, enc_output, W_q2, W_k2, W_v2, W_o2, num_heads, mask=None)
    norm2 = _layer_norm(norm1 + cross_attn, gamma2, beta2)

    # --- Feed-forward network ---
    ffn_hidden = np.maximum(0, norm2 @ W1.T + b1)
    ffn_output = ffn_hidden @ W2.T + b2

    # --- Add & LayerNorm 3 ---
    output = _layer_norm(norm2 + ffn_output, gamma3, beta3)

    return {
        "self_attn": np.round(self_attn, 4).tolist(),
        "norm1": np.round(norm1, 4).tolist(),
        "cross_attn": np.round(cross_attn, 4).tolist(),
        "norm2": np.round(norm2, 4).tolist(),
        "ffn_output": np.round(ffn_output, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }