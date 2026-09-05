def flop_estimator(matmuls, attention_flops=0):
    """
    Returns: dictionary containing exact forward, backward, and total FLOP counts
    """
    forward_flops  = sum(2 * B * D * K for B, D, K in matmuls) + attention_flops
    backward_flops = 2 * forward_flops
    total_flops    = forward_flops + backward_flops

    return {
        "forward_flops":  forward_flops,
        "backward_flops": backward_flops,
        "total_flops":    total_flops,
    }