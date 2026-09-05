def cache_aware_prefill_routing(request_token_counts, cached_prefix_lengths,
                                worker_ids, worker_capacities):
    """
    Returns: dictionary containing assignments and worker loads
    """
    W = len(worker_ids)

    # Per-worker tracking (indexed by position, not worker_id)
    assigned_tokens  = [0] * W    # total tokens assigned (capacity check)
    uncached_loads   = [0] * W    # uncached prefill tokens accumulated

    assignments = []

    for req_idx, total_tokens in enumerate(request_token_counts):
        prefix_lens = cached_prefix_lengths[req_idx]   # one per worker position

        best_pos = None
        best_key = None

        for pos in range(W):
            # Feasibility: full request must fit within capacity
            if assigned_tokens[pos] + total_tokens > worker_capacities[pos]:
                continue

            cached     = prefix_lens[pos]
            uncached   = total_tokens - cached
            new_load   = uncached_loads[pos] + uncached
            remaining  = worker_capacities[pos] - assigned_tokens[pos] - total_tokens

            # Rank: longest cached prefix (desc), lowest new uncached load (asc),
            #       greatest remaining capacity (desc), lowest worker_id (asc)
            key = (-cached, new_load, -remaining, worker_ids[pos])

            if best_key is None or key < best_key:
                best_key = key
                best_pos = pos

        if best_pos is None:
            assignments.append(-1)
        else:
            assignments.append(worker_ids[best_pos])
            assigned_tokens[best_pos] += total_tokens
            uncached_loads[best_pos]  += total_tokens - prefix_lens[best_pos]

    return {
        "assignments":        assignments,
        "worker_uncached_loads": uncached_loads,
    }