import numpy as np

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Returns: Dict with "output" and "attention_weights", rounded to 4 decimal places.
    """
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    V = np.array(V, dtype=float)
    W_q = np.array(W_q, dtype=float)
    W_k = np.array(W_k, dtype=float)
    W_v = np.array(W_v, dtype=float)
    W_o = np.array(W_o, dtype=float)

    seq_q, d_model = Q.shape
    seq_k = K.shape[0]
    h = num_heads
    d_head = d_model // h

    # Project inputs
    Q_proj = Q @ W_q.T
    K_proj = K @ W_k.T
    V_proj = V @ W_v.T

    # Split into heads: (seq, d_model) -> (h, seq, d_head)
    def split_heads(x, seq):
        return x.reshape(seq, h, d_head).transpose(1, 0, 2)

    Qh = split_heads(Q_proj, seq_q)
    Kh = split_heads(K_proj, seq_k)
    Vh = split_heads(V_proj, seq_k)

    mask_arr = np.array(mask, dtype=bool) if mask is not None else None

    head_outputs = []
    attn_weights_list = []

    for hi in range(h):
        Qi, Ki, Vi = Qh[hi], Kh[hi], Vh[hi]

        scores = (Qi @ Ki.T) / np.sqrt(d_head)

        if mask_arr is not None:
            scores = np.where(mask_arr, scores, -1e9)

        # Numerically stable softmax
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        head_outputs.append(weights @ Vi)
        attn_weights_list.append(weights)

    # Concatenate heads along feature dim: (h, seq, d_head) -> (seq, h*d_head)
    concat = np.concatenate(head_outputs, axis=-1)

    output = concat @ W_o.T
    attention_weights = np.array(attn_weights_list)

    return {
        "output": np.round(output, 4).tolist(),
        "attention_weights": np.round(attention_weights, 4).tolist(),
    }