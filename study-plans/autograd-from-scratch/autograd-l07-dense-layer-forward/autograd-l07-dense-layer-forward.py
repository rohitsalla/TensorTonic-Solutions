import torch

def dense_layer_forward(inputs, weight_matrix, biases, nonlinear):
    """
    Returns: one layer output tensor in neuron order
    """
    # weight_matrix @ inputs: (output_width, input_width) @ (input_width,) -> (output_width,)
    a = weight_matrix @ inputs + biases
    return torch.tanh(a) if nonlinear else a