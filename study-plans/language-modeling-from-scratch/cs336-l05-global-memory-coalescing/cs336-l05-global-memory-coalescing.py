import numpy as np

def coalescing_analysis(byte_addresses, access_width_bytes=4, cache_line_bytes=128):
    """
    Returns: dictionary containing line IDs, transaction count, and useful-byte fraction
    """
    addrs = np.asarray(byte_addresses, dtype=np.int64)
    W, L  = access_width_bytes, cache_line_bytes

    touched_lines  = set()
    requested_bytes = set()

    for a in addrs:
        a = int(a)
        # Cache lines spanned by this lane's access
        start_line = a // L
        end_line   = (a + W - 1) // L
        for line in range(start_line, end_line + 1):
            touched_lines.add(line)

        # Every byte requested by this lane (union across lanes)
        for byte_offset in range(W):
            requested_bytes.add(a + byte_offset)

    line_ids          = np.array(sorted(touched_lines), dtype=np.int64)
    transaction_count = int(len(touched_lines))
    C                 = transaction_count * L
    U                 = len(requested_bytes)
    useful_byte_fraction = float(U) / float(C)

    return {
        "line_ids":            line_ids,
        "transaction_count":   transaction_count,
        "useful_byte_fraction": useful_byte_fraction,
    }