import math

def memory_accountant(param_shapes, param_bytes_per_element, grad_bytes_per_element,
                      activation_shapes, activation_bytes_per_element,
                      optimizer, optimizer_bytes_per_element):
    """
    Returns: dictionary containing exact parameter, gradient, activation, optimizer, and total bytes
    """
    # Total parameter elements across all param tensors
    param_elements = sum(math.prod(shape) for shape in param_shapes)

    # Total activation elements across all activation tensors
    activation_elements = sum(math.prod(shape) for shape in activation_shapes)

    # Optimizer state tensors per parameter element: sgd=0, adagrad=1, adam=2
    optimizer_multiplier = {"sgd": 0, "adagrad": 1, "adam": 2}[optimizer]

    parameters     = param_elements * param_bytes_per_element
    gradients      = param_elements * grad_bytes_per_element
    activations    = activation_elements * activation_bytes_per_element
    optimizer_state = param_elements * optimizer_multiplier * optimizer_bytes_per_element
    total          = parameters + gradients + activations + optimizer_state

    return {
        "parameters":     parameters,
        "gradients":      gradients,
        "activations":    activations,
        "optimizer_state": optimizer_state,
        "total":          total,
    }