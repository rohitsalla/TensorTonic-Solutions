import torch
import torch.nn.functional as F

def compact_policy_value_network(x, trunk_weight, trunk_scale, trunk_shift,
                                  residual_parameters, head_parameters, epsilon):
    """
    Returns: raw policy logits and tanh-bounded values.
    """
    def to_f32(t):
        return torch.as_tensor(t, dtype=torch.float32, device=x.device)

    def batch_norm(t, scale, shift, eps):
        mean = t.mean(dim=(0, 2, 3), keepdim=True)
        var  = t.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        t_n  = (t - mean) / (var + eps).sqrt()
        return t_n * to_f32(scale).reshape(1, -1, 1, 1) \
                   + to_f32(shift).reshape(1, -1, 1, 1)

    # ── Trunk: 3×3 conv + BN + ReLU ──────────────────────────────────────
    H = x.float()
    H = F.conv2d(H, to_f32(trunk_weight), bias=None, stride=1, padding=1)
    H = batch_norm(H, trunk_scale, trunk_shift, epsilon)
    H = F.relu(H)

    # ── Residual blocks ───────────────────────────────────────────────────
    for blk in residual_parameters:
        # First conv + BN + ReLU
        R = F.conv2d(H, to_f32(blk['conv1_weight']), bias=None, stride=1, padding=1)
        R = batch_norm(R, blk['scale1'], blk['shift1'], epsilon)
        R = F.relu(R)
        # Second conv + BN
        R = F.conv2d(R, to_f32(blk['conv2_weight']), bias=None, stride=1, padding=1)
        R = batch_norm(R, blk['scale2'], blk['shift2'], epsilon)
        # Shortcut + ReLU
        H = F.relu(H + R)

    # ── Policy head ───────────────────────────────────────────────────────
    hp = head_parameters
    p = F.conv2d(H, to_f32(hp['policy_conv_weight']), bias=None, stride=1, padding=0)
    p = batch_norm(p, hp['policy_scale'], hp['policy_shift'], epsilon)
    p = F.relu(p)
    p = p.flatten(start_dim=1)
    logits = F.linear(p, to_f32(hp['policy_linear_weight']),
                         to_f32(hp['policy_linear_bias']))

    # ── Value head ────────────────────────────────────────────────────────
    v = F.conv2d(H, to_f32(hp['value_conv_weight']), bias=None, stride=1, padding=0)
    v = batch_norm(v, hp['value_scale'], hp['value_shift'], epsilon)
    v = F.relu(v)
    v = v.flatten(start_dim=1)
    v = F.relu(F.linear(v, to_f32(hp['value_hidden_weight']),
                           to_f32(hp['value_hidden_bias'])))
    v = torch.tanh(F.linear(v, to_f32(hp['value_output_weight']),
                               to_f32(hp['value_output_bias']))).squeeze(-1)

    return (logits, v)