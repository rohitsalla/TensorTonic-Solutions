import numpy as np

def seq2seq_forward(src, tgt, W_enc_embed, W_dec_embed, W_enc, b_enc, W_dec, b_dec, W_out, b_out):
    """
    Returns: Dict with "encoder_hiddens", "decoder_hiddens", and "logits", values rounded to 4 decimals.
    """
    W_enc_embed = np.array(W_enc_embed, dtype=float)
    W_dec_embed = np.array(W_dec_embed, dtype=float)
    W_enc = np.array(W_enc, dtype=float)
    b_enc = np.array(b_enc, dtype=float)
    W_dec = np.array(W_dec, dtype=float)
    b_dec = np.array(b_dec, dtype=float)
    W_out = np.array(W_out, dtype=float)
    b_out = np.array(b_out, dtype=float)

    H = b_enc.shape[0]

    # --- Encoder ---
    h = np.zeros(H)
    encoder_hiddens = []
    for token in src:
        x = W_enc_embed[token]
        z = np.concatenate([h, x])  # [h_{t-1}; x_t]
        h = np.tanh(W_enc @ z + b_enc)
        encoder_hiddens.append(h)

    # --- Decoder (initialized with encoder's final hidden state) ---
    h_dec = encoder_hiddens[-1] if encoder_hiddens else np.zeros(H)
    decoder_hiddens = []
    logits = []
    for token in tgt:
        x = W_dec_embed[token]
        z = np.concatenate([h_dec, x])  # [h_{t-1}; x_t]
        h_dec = np.tanh(W_dec @ z + b_dec)
        decoder_hiddens.append(h_dec)
        logit = W_out @ h_dec + b_out
        logits.append(logit)

    return {
        "encoder_hiddens": np.round(np.array(encoder_hiddens), 4).tolist(),
        "decoder_hiddens": np.round(np.array(decoder_hiddens), 4).tolist(),
        "logits": np.round(np.array(logits), 4).tolist(),
    }