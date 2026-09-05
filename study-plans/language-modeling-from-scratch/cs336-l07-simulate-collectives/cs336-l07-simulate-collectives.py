import torch

def simulate_collectives(rank_tensors, collective):
    """
    Returns: list containing one output tensor per rank
    """
    R = len(rank_tensors)

    if collective == "all_gather":
        # Every rank gets the full concatenation of all rank tensors
        gathered = torch.cat([t.clone() for t in rank_tensors])
        return [gathered.clone() for _ in range(R)]

    elif collective == "all_reduce":
        # Every rank gets the elementwise sum across all ranks
        reduced = torch.stack([t.float() for t in rank_tensors]).sum(dim=0).to(rank_tensors[0].dtype)
        return [reduced.clone() for _ in range(R)]

    elif collective == "reduce_scatter":
        # All-reduce first, then give chunk r to rank r
        reduced = torch.stack([t.float() for t in rank_tensors]).sum(dim=0).to(rank_tensors[0].dtype)
        chunks  = torch.chunk(reduced, R)
        return [chunk.clone() for chunk in chunks]

    elif collective == "all_to_all":
        # Split each rank's tensor into R chunks; rank r gets chunk r from every source
        split = [torch.chunk(t, R) for t in rank_tensors]  # split[src][dst_rank]
        return [torch.cat([split[src][dst].clone() for src in range(R)])
                for dst in range(R)]

    else:
        raise ValueError(f"Unknown collective: {collective!r}")