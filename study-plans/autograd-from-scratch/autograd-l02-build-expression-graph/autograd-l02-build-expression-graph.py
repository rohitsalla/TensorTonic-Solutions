import numpy as np

def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """
    nodes = []        # all records in creation order
    lookup = {}       # id -> node record for fast parent lookup

    # Create leaf nodes
    for leaf in leaves:
        node = {
            'id':      leaf['id'],
            'data':    float(np.float64(leaf['data'])),
            'grad':    0.0,
            'op':      '',
            'parents': [],
        }
        nodes.append(node)
        lookup[node['id']] = node

    # Create operation nodes
    for op_rec in operations:
        left_node  = lookup[op_rec['left']]
        right_node = lookup[op_rec['right']]
        op         = op_rec['op']

        lv = np.float64(left_node['data'])
        rv = np.float64(right_node['data'])
        data = float(lv + rv if op == '+' else lv * rv)

        node = {
            'id':      op_rec['id'],
            'data':    data,
            'grad':    0.0,
            'op':      op,
            'parents': [op_rec['left'], op_rec['right']],   # store IDs, not records
        }
        nodes.append(node)
        lookup[node['id']] = node

    final_id = nodes[-1]['id']
    return (nodes, final_id)