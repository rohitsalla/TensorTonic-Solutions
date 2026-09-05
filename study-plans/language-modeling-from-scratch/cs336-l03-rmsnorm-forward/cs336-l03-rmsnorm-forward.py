import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    # Compute mean square in float32 for numerical stability
    x32 = x.float()
    rms = (x32.pow(2).mean(dim=-1, keepdim=True) + epsilon).sqrt()

    # Normalize, apply learned scale, then cast back to input dtype
    return (x32 / rms * g).to(dtype=x.dtype)