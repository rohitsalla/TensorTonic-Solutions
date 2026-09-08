import torch
import torch.nn.functional as F

def policy_value_heads(features, policy_conv_weight, policy_scale, policy_shift,
                       policy_linear_weight, policy_linear_bias,
                       value_conv_weight, value_scale, value_shift,
                       value_hidden_weight, value_hidden_bias,
                       value_output_weight, value_output_bias, epsilon):
    """
    Returns: raw policy logits and one tanh-bounded value per batch item.
    """
    F_in = features.float()

    def batch_norm(t, scale, shift, eps):
        mean = t.mean(dim=(0, 2, 3), keepdim=True)
        var  = t.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        t_n  = (t - mean) / (var + eps).sqrt()
        return t_n * scale.float().reshape(1, -1, 1, 1) \
                   + shift.float().reshape(1, -1, 1, 1)

    # ── Policy head ───────────────────────────────────────────────────────
    # 1×1 conv (2 output channels), BN, ReLU
    p = F.conv2d(F_in, policy_conv_weight.float(), bias=None, stride=1, padding=0)
    p = batch_norm(p, policy_scale, policy_shift, epsilon)
    p = F.relu(p)
    # Flatten spatial + channel dims: (B, 2, H, W) -> (B, 2*H*W)
    p = p.flatten(start_dim=1)
    # Linear layer: (B, 2*H*W) -> (B, actions)
    logits = F.linear(p, policy_linear_weight.float(), policy_linear_bias.float())

    # ── Value head ────────────────────────────────────────────────────────
    # 1×1 conv (1 output channel), BN, ReLU
    v = F.conv2d(F_in, value_conv_weight.float(), bias=None, stride=1, padding=0)
    v = batch_norm(v, value_scale, value_shift, epsilon)
    v = F.relu(v)
    # Flatten: (B, 1, H, W) -> (B, H*W)
    v = v.flatten(start_dim=1)
    # Hidden linear + ReLU
    v = F.relu(F.linear(v, value_hidden_weight.float(), value_hidden_bias.float()))
    # Output linear + tanh -> (B, 1) -> squeeze to (B,)
    v = torch.tanh(F.linear(v, value_output_weight.float(), value_output_bias.float()))
    v = v.squeeze(-1)

    return (logits, v)