def _timeline_metrics(intervals, sm_count):
    per_sm = [[] for _ in range(sm_count)]
    for sm_id, start, end in intervals:
        if end > start:
            per_sm[sm_id].append((float(start), float(end)))
    merged_by_sm = []
    for processor_intervals in per_sm:
        merged = []
        for start, end in sorted(processor_intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        merged_by_sm.append(merged)
    makespan = max(
        (end for merged in merged_by_sm for _, end in merged), default=0.0
    )
    busy_time = sum(
        end - start for merged in merged_by_sm for start, end in merged
    )
    utilization = busy_time / (sm_count * makespan) if makespan else 0.0
    idle_tails = [
        makespan - merged[-1][1] if merged else makespan
        for merged in merged_by_sm
    ]
    return {
        "makespan": makespan,
        "busy_time": busy_time,
        "utilization": utilization,
        "idle_tails": idle_tails,
    }

def kernel_timeline_utilization(separate_intervals, fused_intervals,
                                separate_sm_count, fused_sm_count):
    """
    Returns: dictionary containing merged timeline metrics and speedup
    """
    separate = _timeline_metrics(separate_intervals, separate_sm_count)
    fused = _timeline_metrics(fused_intervals, fused_sm_count)
    speedup = (
        1.0
        if fused["makespan"] == 0
        else separate["makespan"] / fused["makespan"]
    )
    return {
        "separate": separate,
        "fused": fused,
        "relative_makespan_speedup": speedup,
    }
