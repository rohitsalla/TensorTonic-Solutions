import torch

def squared_error_loss_gradients(predictions, targets):
    """
    Returns: the summed squared-error loss and prediction gradients
    """
    residuals = predictions - targets
    loss      = (residuals ** 2).sum()
    grad      = 2 * residuals
    return (loss, grad)