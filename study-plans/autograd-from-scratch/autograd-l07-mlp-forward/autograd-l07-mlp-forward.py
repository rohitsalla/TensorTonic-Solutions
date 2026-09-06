import torch

def mlp_forward(inputs, weights, biases):
    """
    Returns: the final output tensor and all layer output tensors in forward order
    """
    h = inputs
    layer_outputs = []

    for W, b in zip(weights, biases):
        h = torch.tanh(W @ h + b)
        layer_outputs.append(h)

    return (h, layer_outputs)