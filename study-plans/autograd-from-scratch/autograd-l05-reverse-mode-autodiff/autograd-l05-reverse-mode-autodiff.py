import numpy as np

def reverse_mode_autodiff(leaves, operations, output_id):
    """
    Returns: (output_value, gradients) with one gradient for every reachable leaf ID.
    """
    # Build lookup: id -> record, and track which IDs are leaves
    all_nodes = {}
    leaf_ids  = set()
    for leaf in leaves:
        all_nodes[leaf['id']] = {'id': leaf['id'], 'value': np.float64(leaf['value']),
                                  'op': None, 'parents': []}
        leaf_ids.add(leaf['id'])
    for op in operations:
        all_nodes[op['id']] = {'id': op['id'], 'value': None,
                                'op': op['op'], 'parents': op['parents']}

    # Step 1: Find reachable nodes via backward DFS from output_id
    reachable = set()
    def mark_reachable(nid):
        if nid in reachable: return
        reachable.add(nid)
        for pid in all_nodes[nid]['parents']:
            mark_reachable(pid)
    mark_reachable(output_id)

    # Step 2: Topological sort (Kahn's + min-heap for input-order tie-breaking)
    import heapq
    # Combine leaves then operations to get input order
    all_input_records = list(leaves) + list(operations)
    input_order = {rec['id']: i for i, rec in enumerate(all_input_records)}

    in_degree = {nid: 0 for nid in reachable}
    for nid in reachable:
        for pid in all_nodes[nid]['parents']:
            if pid in reachable:
                in_degree[nid] += 1

    heap = [(input_order[nid], nid) for nid in reachable if in_degree[nid] == 0]
    heapq.heapify(heap)
    topo = []

    while heap:
        _, nid = heapq.heappop(heap)
        topo.append(nid)
        for cid in reachable:
            if nid in all_nodes[cid]['parents']:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    heapq.heappush(heap, (input_order[cid], cid))

    # Step 3: Forward pass — evaluate values in topological order
    for nid in topo:
        node = all_nodes[nid]
        if node['op'] is None:
            pass   # leaf: value already set
        elif node['op'] == 'add':
            a, b = node['parents']
            node['value'] = all_nodes[a]['value'] + all_nodes[b]['value']
        elif node['op'] == 'mul':
            a, b = node['parents']
            node['value'] = all_nodes[a]['value'] * all_nodes[b]['value']
        elif node['op'] == 'tanh':
            p = node['parents'][0]
            node['value'] = np.tanh(all_nodes[p]['value'])

    output_value = float(all_nodes[output_id]['value'])

    # Step 4: Backward pass — accumulate gradients in reverse topological order
    grads = {nid: np.float64(0.0) for nid in reachable}
    grads[output_id] = np.float64(1.0)   # seed

    for nid in reversed(topo):
        node = all_nodes[nid]
        g = grads[nid]
        if node['op'] is None:
            pass   # leaf: no children to push to
        elif node['op'] == 'add':
            a, b = node['parents']
            grads[a] += g
            grads[b] += g
        elif node['op'] == 'mul':
            a, b = node['parents']
            grads[a] += g * all_nodes[b]['value']
            grads[b] += g * all_nodes[a]['value']
        elif node['op'] == 'tanh':
            p = node['parents'][0]
            t = node['value']   # saved forward output
            grads[p] += g * (1 - t * t)

    # Step 5: Collect leaf gradients in original leaf input order
    leaf_grads = {leaf['id']: float(grads[leaf['id']])
                  for leaf in leaves if leaf['id'] in reachable}

    return (output_value, leaf_grads)