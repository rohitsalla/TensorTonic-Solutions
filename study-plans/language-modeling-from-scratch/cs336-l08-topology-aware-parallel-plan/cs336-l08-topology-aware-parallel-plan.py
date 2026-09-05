def topology_aware_parallel_plan(num_gpus, gpus_per_node, model_state_bytes,
                                 activation_bytes_per_sample, global_batch_size,
                                 sequence_length, memory_per_gpu,
                                 min_context_parallel):
    """
    Returns: best feasible plan dictionary or None
    """
    def ceil_div(a, b):
        return (a + b - 1) // b

    def divisors(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    G    = num_gpus
    divs = divisors(G)

    best     = None
    best_key = None

    for d in divs:
        if global_batch_size % d != 0:          # batch must divide evenly
            continue
        for t in divs:
            for p in divs:
                for c in divs:
                    # Must use every GPU exactly once
                    if d * t * p * c != G:
                        continue
                    # Sequence must divide evenly across context-parallel ranks
                    if sequence_length % c != 0:
                        continue
                    # Minimum context parallelism
                    if c < min_context_parallel:
                        continue
                    # Tensor × context must fit within one node (high-bandwidth domain)
                    if t * c > gpus_per_node:
                        continue

                    # Memory estimates (integer ceiling division)
                    Ms = ceil_div(model_state_bytes, t * p)
                    Ma = ceil_div(activation_bytes_per_sample * (global_batch_size // d), t * c)

                    if Ms + Ma > memory_per_gpu:
                        continue

                    # Communication scores
                    I_out = 2 * Ms * (d - 1) / d + Ma * (p - 1) / p
                    I_in  = 2 * Ma * ((t - 1) / t + (c - 1) / c)

                    # Lexicographic sort key: minimize (I_out, I_in, p, -d, t, c)
                    key = (I_out, I_in, p, -d, t, c)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = {
                            "data_parallel":       d,
                            "tensor_parallel":     t,
                            "pipeline_parallel":   p,
                            "context_parallel":    c,
                            "estimated_memory_bytes": Ms + Ma,
                            "inter_node_bytes":    float(I_out),
                            "intra_node_bytes":    float(I_in),
                        }

    return best