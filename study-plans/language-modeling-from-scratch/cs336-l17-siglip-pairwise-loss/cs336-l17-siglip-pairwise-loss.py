import torch
import torch.nn.functional as F

def siglip_pairwise_loss(image_embeddings, text_embeddings, logit_scale, bias):
    """
    Returns: dictionary containing SigLIP loss, logits, and pair labels
    """
    # L2-normalize both embedding matrices row-wise
    v = F.normalize(image_embeddings, p=2, dim=-1)
    t = F.normalize(text_embeddings,  p=2, dim=-1)

    # Scaled and biased logits: ℓ_ij = a * v̂_i · t̂_j + b
    logits = logit_scale * (v @ t.T) + bias        # (B, B)

    # Labels: +1 on diagonal, -1 off-diagonal
    B          = logits.shape[0]
    pair_labels = torch.full((B, B), -1.0,
                             dtype=logits.dtype, device=logits.device)
    pair_labels.fill_diagonal_(1.0)

    # Loss: -1/B² Σ log σ(z_ij * ℓ_ij)
    # log σ(x) = F.logsigmoid(x) for numerical stability
    loss = -(F.logsigmoid(pair_labels * logits)).mean()

    return {
        "loss":        loss,
        "logits":      logits,
        "pair_labels": pair_labels,
    }