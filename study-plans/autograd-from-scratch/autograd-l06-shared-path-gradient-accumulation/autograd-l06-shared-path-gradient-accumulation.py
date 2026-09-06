import numpy as np
import heapq

def shared_path_gradient_accumulation(leaves, operations, output_id):
    """
    Returns: one accumulated gradient for every reachable leaf ID.
    """
    # ── Build node lookup ─────────────────────────────────────────────────
    nodes = {}
    for leaf in leaves:
        nodes[leaf['id']] = {'value': np.float64(leaf['value']), 'op': None, 'parents': []}
    for op in operations:
        nodes[op['id']] = {'value': None, 'op': op['op'], 'parents': op['parents']}

    # ── Find reachable nodes via DFS ──────────────────────────────────────
    reachable = set()
    def dfs(nid):
        if nid in reachable: return
        reachable.add(nid)
        for pid in nodes[nid]['parents']: dfs(pid)
    dfs(output_id)

    # ── Topological sort (Kahn's + heap for input-order tie-breaking) ─────
    all_records = list(leaves) + list(operations)
    input_order = {rec['id']: i for i, rec in enumerate(all_records)}

    # In-degree counts ALL parent-slot occurrences (a node used twice counts twice)
    in_degree = {nid: 0 for nid in reachable}
    for nid in reachable:
        for pid in nodes[nid]['parents']:
            if pid in reachable:
                in_degree[nid] += 1

    heap = [(input_order[nid], nid) for nid in reachable if in_degree[nid] == 0]
    heapq.heapify(heap)
    topo = []
    while heap:
        _, nid = heapq.heappop(heap)
        topo.append(nid)
        # Decrement in-degree for each child that lists nid as a parent
        for cid in reachable:
            count = nodes[cid]['parents'].count(nid)
            if count:
                in_degree[cid] -= count
                if in_degree[cid] == 0:
                    heapq.heappush(heap, (input_order[cid], cid))

    # ── Forward pass ──────────────────────────────────────────────────────
    for nid in topo:
        node = nodes[nid]
        if node['op'] is None: pass
        elif node['op'] == 'add':
            a, b = node['parents']
            node['value'] = nodes[a]['value'] + nodes[b]['value']
        elif node['op'] == 'mul':
            a, b = node['parents']
            node['value'] = nodes[a]['value'] * nodes[b]['value']
        elif node['op'] == 'tanh':
            node['value'] = np.tanh(nodes[node['parents'][0]]['value'])

    # ── Backward pass — += accumulates across all uses ────────────────────
    grads = {nid: np.float64(0.0) for nid in reachable}
    grads[output_id] = np.float64(1.0)

    for nid in reversed(topo):
        g = grads[nid]; node = nodes[nid]
        if node['op'] == 'add':
            a, b = node['parents']
            grads[a] += g; grads[b] += g
        elif node['op'] == 'mul':
            a, b = node['parents']
            grads[a] += g * nodes[b]['value']
            grads[b] += g * nodes[a]['value']
        elif node['op'] == 'tanh':
            t = node['value']
            grads[node['parents'][0]] += g * (1 - t * t)

    return {leaf['id']: float(grads[leaf['id']])
            for leaf in leaves if leaf['id'] in reachable}