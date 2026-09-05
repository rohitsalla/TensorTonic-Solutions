def kv_cache_capacity_planner(model_parameters, bytes_per_parameter, num_layers,
                              sequence_length, num_attention_heads, num_kv_heads,
                              head_dim, bytes_per_element, memory_per_gpu,
                              reserved_memory, memory_bandwidth):
    """
    Returns: dictionary containing cache capacity and bandwidth metrics
    """
    # ── Integer byte calculations ─────────────────────────────────────────
    M_model   = model_parameters * bytes_per_parameter
    M_request = 2 * num_layers * sequence_length * num_kv_heads * head_dim * bytes_per_element

    available = memory_per_gpu - reserved_memory - M_model
    B = max(0, available // M_request) if M_request > 0 else 0

    # ── Floating-point bandwidth and throughput ───────────────────────────
    total_bytes      = M_model + B * M_request
    seconds_per_token = total_bytes / memory_bandwidth

    tokens_per_second = 0.0 if B == 0 else B / seconds_per_token

    return {
        "model_bytes":          M_model,
        "kv_bytes_per_request": M_request,
        "max_batch_size":       B,
        "seconds_per_token":    float(seconds_per_token),
        "tokens_per_second":    float(tokens_per_second),
    }