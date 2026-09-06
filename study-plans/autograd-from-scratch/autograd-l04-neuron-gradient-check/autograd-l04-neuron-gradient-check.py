import torch

def neuron_gradient_check(inputs, weights, bias, h):
    """
    Returns: analytic and numerical parameter gradients with their maximum error
    """
    device = inputs.device
    x = inputs.to(torch.float64)
    w = weights.to(torch.float64)
    b = bias.to(torch.float64)
    h = torch.tensor(h, dtype=torch.float64)

    def neuron(x, w, b):
        return torch.tanh(torch.dot(x, w) + b)

    y = neuron(x, w, b)
    local_slope = 1.0 - y * y

    # Analytic gradients
    analytic_weight_gradients = local_slope * x
    analytic_bias_gradient    = local_slope

    # Numerical weight gradients
    n = len(w)
    numerical_weight_gradients = torch.zeros(n, dtype=torch.float64, device=device)
    for i in range(n):
        w_perturbed = w.clone()
        w_perturbed[i] = w_perturbed[i] + h
        numerical_weight_gradients[i] = (neuron(x, w_perturbed, b) - y) / h

    # Numerical bias gradient
    numerical_bias_gradient = (neuron(x, w, b + h) - y) / h

    # Maximum absolute disagreement — return as float64 tensor, not Python float
    all_analytic  = torch.cat([analytic_weight_gradients,  analytic_bias_gradient.unsqueeze(0)])
    all_numerical = torch.cat([numerical_weight_gradients, numerical_bias_gradient.unsqueeze(0)])
    max_error = torch.max(torch.abs(all_analytic - all_numerical))   # scalar tensor

    return (
        analytic_weight_gradients,
        numerical_weight_gradients,
        analytic_bias_gradient,
        numerical_bias_gradient,
        max_error,   # torch.float64 scalar tensor, not float()
    )