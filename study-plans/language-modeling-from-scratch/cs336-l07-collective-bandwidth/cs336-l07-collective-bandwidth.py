def collective_bandwidth(payload_bytes, world_size, duration_seconds, collective):
    """
    Returns: dictionary containing algorithm bytes and bandwidth
    """
    c = 2 if collective == "all_reduce" else 1

    algorithm_bytes           = c * payload_bytes * (world_size - 1) / world_size
    bandwidth_bytes_per_second = algorithm_bytes / duration_seconds

    return {
        "algorithm_bytes":            float(algorithm_bytes),
        "bandwidth_bytes_per_second": float(bandwidth_bytes_per_second),
    }