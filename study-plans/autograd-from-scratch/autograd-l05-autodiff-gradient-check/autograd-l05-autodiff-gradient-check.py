import numpy as np
import heapq

def autodiff_gradient_check(leaves, operations, output_id, h):
    """
    Returns: (analytic_gradients, numerical_gradients, max_error) keyed by reachable leaf ID.
    """
    h = np.float64(h)

    # ── Build node lookup ─────────────────────────────────────────────────
    def build_lookup(leaves):
        nodes = {}
        for leaf in leaves:
            nodes[leaf['id']] = {'id': leaf['id'], 'value': np.float64(leaf['value']),
                                  'op': None, 'parents': []}
        for op in operations:
            nodes[op['id']] = {'id': op['id'], 'value': None,
                                'op': op['op'], 'parents': op['parents']}
        return nodes

    # ── Find reachable nodes ──────────────────────────────────────────────
    def find_reachable(nodes, output_id):
        reachable = set()
        def dfs(nid):
            if nid in reachable: return
            reachable.add(nid)
            for pid in nodes[nid]['parents']: dfs(pid)
        dfs(output_id)
        return reachable

    # ── Topological sort (Kahn's + heap for input-order tie-breaking) ─────
    def topo_sort(nodes, reachable, input_order):
        in_deg = {nid: 0 for nid in reachable}
        for nid in reachable:
            for pid in nodes[nid]['parents']:
                if pid in reachable: in_deg[nid] += 1
        heap = [(input_order[nid], nid) for nid in reachable if in_deg[nid] == 0]
        heapq.heapify(heap)
        topo = []
        while heap:
            _, nid = heapq.heappop(heap)
            topo.append(nid)
            for cid in reachable:
                if nid in nodes[cid]['parents']:
                    in_deg[cid] -= 1
                    if in_deg[cid] == 0:
                        heapq.heappush(heap, (input_order[cid], cid))
        return topo

    # ── Forward evaluation ────────────────────────────────────────────────
    def forward(nodes, topo):
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
        return nodes[topo[-1]]['value']   # last in topo = output (since output_id is the root)

    # ── Reverse pass ──────────────────────────────────────────────────────
    def backward(nodes, topo, output_id):
        grads = {nid: np.float64(0.0) for nid in topo}
        grads[output_id] = np.float64(1.0)
        for nid in reversed(topo):
            g = grads[nid]; node = nodes[nid]
            if node['op'] == 'add':
                a, b = node['parents']; grads[a] += g; grads[b] += g
            elif node['op'] == 'mul':
                a, b = node['parents']
                grads[a] += g * nodes[b]['value']
                grads[b] += g * nodes[a]['value']
            elif node['op'] == 'tanh':
                t = node['value']
                grads[node['parents'][0]] += g * (1 - t * t)
        return grads

    # ── Setup ─────────────────────────────────────────────────────────────
    all_input_records = list(leaves) + list(operations)
    input_order = {rec['id']: i for i, rec in enumerate(all_input_records)}

    nodes     = build_lookup(leaves)
    reachable = find_reachable(nodes, output_id)
    topo      = topo_sort(nodes, reachable, input_order)

    # Ensure output_id is last in topo for forward() to return correctly
    # (re-find output value after forward pass instead)
    forward(nodes, topo)
    f0 = nodes[output_id]['value']

    grads     = backward(nodes, topo, output_id)

    # ── Analytic gradients for reachable leaves ───────────────────────────
    reachable_leaves = [leaf for leaf in leaves if leaf['id'] in reachable]
    analytic = {leaf['id']: float(grads[leaf['id']]) for leaf in reachable_leaves}

    # ── Numerical gradients: perturb one leaf at a time ───────────────────
    numerical = {}
    for leaf in reachable_leaves:
        lid = leaf['id']
        # Build fresh graph with one leaf perturbed
        perturbed_leaves = [
            {'id': l['id'], 'value': np.float64(l['value']) + (h if l['id'] == lid else 0)}
            for l in leaves
        ]
        p_nodes = build_lookup(perturbed_leaves)
        p_topo  = topo_sort(p_nodes, reachable, input_order)
        forward(p_nodes, p_topo)
        fh = p_nodes[output_id]['value']
        numerical[lid] = float((fh - f0) / h)

    # ── Maximum absolute disagreement ─────────────────────────────────────
    max_error = float(max(
        abs(analytic[lid] - numerical[lid]) for lid in analytic
    )) if analytic else 0.0

    return (analytic, numerical, max_error)