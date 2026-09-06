def trace_reachable_graph(nodes, output_id):
    """
    Returns: reachable node IDs and parent-to-child edges in deterministic order
    """
    # Step 1: Find all reachable node IDs via backward BFS/DFS from output_id
    reachable = set()

    def mark_reachable(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        node = node_lookup[node_id]
        for parent_id in node['parents']:
            mark_reachable(parent_id)

    node_lookup = {n['id']: n for n in nodes}
    mark_reachable(output_id)

    # Step 2: Collect reachable IDs in original node-list order
    reachable_ids = [n['id'] for n in nodes if n['id'] in reachable]

    # Step 3: Collect edges in child-input-order, then parent-list order
    # For each reachable node (in original list order), emit [parent_id, child_id]
    # for each parent in its stored order — only if child is reachable
    edges = []
    for node in nodes:
        if node['id'] not in reachable:
            continue
        for parent_id in node['parents']:
            edges.append([parent_id, node['id']])

    return (reachable_ids, edges)