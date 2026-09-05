def zero_memory_accounting(parameter_bytes, gradient_bytes, optimizer_bytes, world_size):
    """
    Returns: dictionary containing DDP and ZeRO byte totals
    """
    P, G, O, R = parameter_bytes, gradient_bytes, optimizer_bytes, world_size

    def ceil_div(x, r):
        return (x + r - 1) // r

    ddp   = P + G + O
    zero1 = P + G + ceil_div(O, R)
    zero2 = P + ceil_div(G + O, R)
    zero3 = ceil_div(P + G + O, R)

    return {"ddp": ddp, "zero1": zero1, "zero2": zero2, "zero3": zero3}