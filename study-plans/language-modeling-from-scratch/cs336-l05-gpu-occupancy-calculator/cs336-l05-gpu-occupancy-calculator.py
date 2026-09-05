import math

def gpu_occupancy(threads_per_block, registers_per_thread, shared_mem_per_block,
                  max_threads_per_sm, max_warps_per_sm, max_blocks_per_sm,
                  max_registers_per_sm, max_shared_mem_per_sm, warp_size=32):
    """
    Returns: dictionary containing resident blocks, resident warps, and occupancy
    """
    # Warps allocated per block (ceiling division)
    warps_per_block   = (threads_per_block + warp_size - 1) // warp_size
    effective_threads = warps_per_block * warp_size

    # A block that exceeds any per-SM limit can't run at all
    if (effective_threads > max_threads_per_sm or
            warps_per_block  > max_warps_per_sm):
        return {"blocks_per_sm": 0, "resident_warps": 0, "occupancy": 0.0}

    # Independent block-count limits from each resource
    limit_threads = max_threads_per_sm // effective_threads
    limit_warps   = max_warps_per_sm   // warps_per_block
    limit_blocks  = max_blocks_per_sm

    # Register limit: zero regs per thread -> nonbinding
    if registers_per_thread > 0:
        total_regs   = effective_threads * registers_per_thread
        limit_regs   = max_registers_per_sm // total_regs
    else:
        limit_regs   = max_blocks_per_sm   # nonbinding: cap at hardware block max

    # Shared-memory limit: zero shared mem -> nonbinding
    if shared_mem_per_block > 0:
        limit_smem   = max_shared_mem_per_sm // shared_mem_per_block
    else:
        limit_smem   = max_blocks_per_sm   # nonbinding

    blocks_per_sm  = min(limit_threads, limit_warps, limit_blocks,
                         limit_regs,   limit_smem)
    resident_warps = blocks_per_sm * warps_per_block
    occupancy      = resident_warps / max_warps_per_sm

    return {
        "blocks_per_sm":  blocks_per_sm,
        "resident_warps": resident_warps,
        "occupancy":      occupancy,
    }