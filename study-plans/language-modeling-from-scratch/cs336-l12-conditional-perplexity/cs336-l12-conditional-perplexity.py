import torch
import math

def conditional_perplexity(logits, target_ids, response_mask):
    """
    Returns: dictionary containing response-token NLL, perplexity, and token count
    """
    # Log-probabilities over vocabulary: (B, S, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Gather log-prob of each target token: (B, S)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Mask to response positions only
    M = int(response_mask.sum())

    if M == 0:
        return {
            "negative_log_likelihood": 0.0,
            "perplexity":              1.0,
            "selected_token_count":    0,
        }

    nll = float(-token_log_probs[response_mask].sum() / M)
    ppl = math.exp(nll)

    return {
        "negative_log_likelihood": nll,
        "perplexity":              ppl,
        "selected_token_count":    M,
    }