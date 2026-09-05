def pipeline_bubble_utilization(num_stages, num_microbatches, include_backward):
    """
    Returns: dictionary containing the schedule and utilization metrics
    """
    p, m = num_stages, num_microbatches
    T    = m + p - 1   # slots per directional wave

    schedule = []

    # ── Forward wave ──────────────────────────────────────────────────────
    for t in range(T):
        slot = []
        for s in range(p):
            i = t - s
            slot.append(f"F{i}" if 0 <= i < m else None)
        schedule.append(slot)

    # ── Backward wave (mirrored) ──────────────────────────────────────────
    if include_backward:
        for u in range(T):
            slot = []
            for s in range(p):
                i = m - 1 - (u - (p - 1 - s))
                slot.append(f"B{i}" if 0 <= i < m else None)
            schedule.append(slot)

    # ── Metrics ───────────────────────────────────────────────────────────
    q           = 2 if include_backward else 1
    useful      = q * p * m
    total_slots = q * T * p
    bubble_slots = total_slots - useful
    utilization  = useful / total_slots

    return {
        "schedule":    schedule,
        "bubble_slots": bubble_slots,
        "total_slots":  total_slots,
        "utilization":  float(utilization),
    }