import torch

def recursive_parameter_collection(weights, biases):
    """
    Returns: scalar parameter views in traversal order and the parameter count
    """
    params = []

    for W, b in zip(weights, biases):
        # W: (out_neurons, in_width), b: (out_neurons,)
        for neuron_idx in range(W.shape[0]):
            # Add each input weight for this neuron in column order
            for col_idx in range(W.shape[1]):
                params.append(W[neuron_idx, col_idx])
            # Then append this neuron's bias
            params.append(b[neuron_idx])

    return (params, len(params))