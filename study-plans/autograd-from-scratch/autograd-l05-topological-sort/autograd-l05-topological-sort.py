import heapq

def topological_sort(nodes, output_id):
    """
    Returns: reachable node IDs in deterministic topological order using input order to break valid ordering ties.
    """
    node_lookup = {n['id']: n for n in nodes}
    input_order = {n['id']: i for i, n in enumerate(nodes)}

    # Step 1: Find reachable nodes via backward DFS
    reachable = set()
    def mark_reachable(nid):
        if nid in reachable: return
        reachable.add(nid)
        for pid in node_lookup[nid]['parents']:
            mark_reachable(pid)
    mark_reachable(output_id)

    # Step 2: In-degrees within reachable subgraph
    in_degree = {nid: 0 for nid in reachable}
    for n in nodes:
        if n['id'] not in reachable: continue
        for pid in n['parents']:
            if pid in reachable:
                in_degree[n['id']] += 1

    # Step 3: Kahn's with min-heap keyed by original input position
    # — guarantees ties go to the node appearing earliest in the input list
    heap = [(input_order[nid], nid) for nid in reachable if in_degree[nid] == 0]
    heapq.heapify(heap)
    result = []

    while heap:
        _, nid = heapq.heappop(heap)
        result.append(nid)
        for n in nodes:
            if n['id'] not in reachable: continue
            if nid in n['parents']:
                in_degree[n['id']] -= 1
                if in_degree[n['id']] == 0:
                    heapq.heappush(heap, (input_order[n['id']], n['id']))

    return result