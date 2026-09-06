import torch

def tanh_forward_backward(x, upstream_gradient):
    """
    Returns: tanh output and its upstream-scaled input gradient
    """
    y = torch.tanh(x)
    input_gradient = upstream_gradient * (1 - y * y)
    return y, input_gradient