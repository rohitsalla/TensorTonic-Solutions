import torch

def multiple_choice_likelihood(logits, token_ids, continuation_mask):
    """
    Returns: dictionary containing continuation scores and selected choices
    """
    # log-softmax over vocabulary dim: (B, C, T, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather target-token log-probs: (B, C, T)
    token_log_probs = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    # Zero out non-continuation positions, then sum over T: (B, C)
    choice_scores = (token_log_probs * continuation_mask.float()).sum(dim=-1)

    # argmax over choices; torch.argmax returns lowest index on ties
    predicted_indices = choice_scores.argmax(dim=-1).to(torch.int64)

    return {
        "choice_scores":     choice_scores,
        "predicted_indices": predicted_indices,
    }