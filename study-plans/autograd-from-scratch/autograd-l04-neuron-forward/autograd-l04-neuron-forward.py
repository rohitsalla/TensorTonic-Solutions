import torch

def neuron_forward(inputs, weights, bias):
    """
    Returns: scalar preactivation and tanh output
    """
    # dot product handles empty tensors correctly (returns 0)
    a = torch.dot(inputs, weights) + bias
    y = torch.tanh(a)
    return a, y