import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    # Forward pass
    a = torch.dot(inputs, weights) + bias
    y = torch.tanh(a)

    # Gradient through tanh: δ = g * (1 - tanh²(a))
    delta = upstream_gradient * (1 - y * y)

    # Chain rule to each parameter
    input_gradients  = delta * weights   # ∂L/∂x_i = δ * w_i
    weight_gradients = delta * inputs    # ∂L/∂w_i = δ * x_i
    bias_gradient    = delta             # ∂L/∂b   = δ

    return y, input_gradients, weight_gradients, bias_gradient