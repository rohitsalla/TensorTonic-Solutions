import numpy as np

def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma + beta

def encoder_block(x, W_q, W_k, W_v, W_o, num_heads, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2):
    """
    Implements one transformer encoder block.
    Returns: Dict with "attention_output", "norm1", "ffn_output", "output",
             all as list-of-lists rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)
    W_q = np.array(W_q, dtype=float); W_k = np.array(W_k, dtype=float)
    W_v = np.array(W_v, dtype=float); W_o = np.array(W_o, dtype=float)
    W1 = np.array(W1, dtype=float); b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float); b2 = np.array(b2, dtype=float)
    gamma1 = np.array(gamma1, dtype=float); beta1 = np.array(beta1, dtype=float)
    gamma2 = np.array(gamma2, dtype=float); beta2 = np.array(beta2, dtype=float)

    seq, d_model = x.shape
    h = num_heads
    d_head = d_model // h

    # --- Multi-head self-attention (Q = K = V = x) ---
    Q = x @ W_q.T
    K = x @ W_k.T
    V = x @ W_v.T

    def split_heads(t):
        return t.reshape(seq, h, d_head).transpose(1, 0, 2)

    Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

    head_outputs = []
    for hi in range(h):
        Qi, Ki, Vi = Qh[hi], Kh[hi], Vh[hi]
        scores = (Qi @ Ki.T) / np.sqrt(d_head)
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        head_outputs.append(weights @ Vi)

    concat = np.concatenate(head_outputs, axis=-1)  # (seq, d_model)
    attn_output = concat @ W_o.T

    # --- Add & LayerNorm ---
    norm1 = _layer_norm(x + attn_output, gamma1, beta1)

    # --- Feed-forward network ---
    ffn_hidden = np.maximum(0, norm1 @ W1.T + b1)
    ffn_output = ffn_hidden @ W2.T + b2

    # --- Add & LayerNorm ---
    output = _layer_norm(norm1 + ffn_output, gamma2, beta2)

    return {
        "attention_output": np.round(attn_output, 4).tolist(),
        "norm1": np.round(norm1, 4).tolist(),
        "ffn_output": np.round(ffn_output, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }