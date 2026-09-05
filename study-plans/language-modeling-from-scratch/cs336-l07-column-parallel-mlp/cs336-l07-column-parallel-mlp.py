import torch

def column_parallel_mlp(x, weight_shards, bias_shards=None):
    """
    Returns: dictionary containing local and full activations
    """
    local_activations = []

    for r, w in enumerate(weight_shards):
        # Y_r = X @ W_r  (batch x D_out_r)
        y_r = x @ w
        if bias_shards is not None:
            y_r = y_r + bias_shards[r]
        local_activations.append(y_r)

    # Concatenate along the feature (last) dimension in rank order
    full_activation = torch.cat(local_activations, dim=-1)

    return {
        "local_activations": local_activations,
        "full_activation":   full_activation,
    }