import torch
import torch.nn.functional as F

def alpha_go_residual_block(x, conv1_weight, scale1, shift1, conv2_weight, scale2, shift2, epsilon):
    """
    Returns: the torch.float32 residual-block output on the input device.
    """
    x = x.float()

    def batch_norm(t, scale, shift, eps):
        # Compute mean and population variance over (batch, height, width) = dims 0,2,3
        mean = t.mean(dim=(0, 2, 3), keepdim=True)                    # (1, C, 1, 1)
        var  = t.var(dim=(0, 2, 3), keepdim=True, unbiased=False)     # population var
        t_norm = (t - mean) / (var + eps).sqrt()
        # Scale and shift: reshape (C,) -> (1, C, 1, 1) for broadcasting
        return t_norm * scale.float().reshape(1, -1, 1, 1) \
                      + shift.float().reshape(1, -1, 1, 1)

    # First conv + BN + ReLU
    H = F.conv2d(x, conv1_weight.float(), bias=None, stride=1, padding=1)
    H = batch_norm(H, scale1, shift1, epsilon)
    H = F.relu(H)

    # Second conv + BN
    Y = F.conv2d(H, conv2_weight.float(), bias=None, stride=1, padding=1)
    Y = batch_norm(Y, scale2, shift2, epsilon)

    # Shortcut + ReLU
    Y = F.relu(x + Y)

    return Y