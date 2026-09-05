import torch
import torch.nn.functional as F

def symmetric_clip_loss(image_embeddings, text_embeddings, temperature):
    """
    Returns: dictionary containing CLIP loss, similarities, and accuracies
    """
    # L2-normalize along feature dimension
    v = F.normalize(image_embeddings, p=2, dim=-1)
    t = F.normalize(text_embeddings,  p=2, dim=-1)

    # Scaled similarity matrix: S_ij = (v_i · t_j) / τ
    S = (v @ t.T) / temperature                        # (N, N)

    # Diagonal targets
    N       = S.shape[0]
    targets = torch.arange(N, device=S.device)

    # Symmetric cross-entropy loss
    loss_i2t = F.cross_entropy(S,   targets)
    loss_t2i = F.cross_entropy(S.T, targets)
    loss     = (loss_i2t + loss_t2i) / 2

    # Top-1 accuracies
    i2t_acc = (S.argmax(dim=-1)   == targets).float().mean()
    t2i_acc = (S.T.argmax(dim=-1) == targets).float().mean()

    return {
        "loss":                   loss,
        "similarities":           S,
        "image_to_text_accuracy": i2t_acc,
        "text_to_image_accuracy": t2i_acc,
    }