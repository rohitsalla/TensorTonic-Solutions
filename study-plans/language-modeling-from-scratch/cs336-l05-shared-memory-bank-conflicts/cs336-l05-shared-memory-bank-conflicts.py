import numpy as np

def bank_conflict_analysis(byte_addresses, num_banks=32, bank_width_bytes=4):
    """
    Returns: dictionary containing bank IDs and conflict degrees
    """
    addrs = np.asarray(byte_addresses, dtype=np.int64)

    # Bank ID for each lane: floor(addr / bank_width) % num_banks
    bank_ids = (addrs // bank_width_bytes) % num_banks   # (N,)

    # Conflict degree: number of distinct addresses in each lane's bank
    conflict_degree = np.ones(len(addrs), dtype=np.int64)

    for bank in np.unique(bank_ids):
        mask        = bank_ids == bank
        lane_addrs  = addrs[mask]
        distinct    = len(np.unique(lane_addrs))    # broadcasts count once
        conflict_degree[mask] = distinct

    return {
        "bank_ids":       bank_ids,
        "conflict_degree": conflict_degree,
    }