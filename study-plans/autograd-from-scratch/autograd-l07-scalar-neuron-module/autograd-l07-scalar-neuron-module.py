import torch

def scalar_neuron_module(inputs, weights, bias, nonlinear):
    """
    Returns: one scalar neuron output tensor
    """
    # torch.dot doesn't support fp16/bf16 on CPU; use (inputs * weights).sum() instead
    a = (inputs * weights).sum() + bias
    return torch.tanh(a) if nonlinear else a