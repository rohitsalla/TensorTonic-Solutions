import torch

def assistant_only_sft_mask(input_ids, role_ids, attention_mask,
                            assistant_role=1):
    """
    Returns: dictionary containing shifted labels and assistant loss mask
    """
    B, S = input_ids.shape

    # Target token at position t is input_ids[t+1], attended by attention_mask[t+1]
    # and belonging to assistant if role_ids[t+1] == assistant_role
    # Source condition: attention_mask[t] must also be True

    labels    = torch.full((B, S), -100, dtype=input_ids.dtype, device=input_ids.device)
    loss_mask = torch.zeros((B, S), dtype=torch.bool,           device=input_ids.device)

    if S < 2:
        return {"labels": labels, "loss_mask": loss_mask}

    # Conditions checked at source position t (predicting target t+1):
    src_attended = attention_mask[:, :-1]                        # (B, S-1)
    tgt_attended = attention_mask[:, 1:]                         # (B, S-1)
    tgt_assistant = role_ids[:, 1:] == assistant_role           # (B, S-1)

    active = src_attended & tgt_attended & tgt_assistant         # (B, S-1)

    # Place target tokens where active, leave -100 elsewhere
    labels[:, :-1]    = torch.where(active, input_ids[:, 1:],
                                    torch.full_like(input_ids[:, 1:], -100))
    loss_mask[:, :-1] = active

    return {"labels": labels, "loss_mask": loss_mask}