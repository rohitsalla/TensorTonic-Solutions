import copy

def backup_mcts_path(path, leaf_value, leaf_player):
    """
    Returns: Updated root-to-leaf path records with incremented N, W, and Q values.
    """
    if not path:
        return []

    result = copy.deepcopy(path)

    # Start at the leaf end, propagating value backward to root
    v = float(leaf_value)
    current_player = leaf_player

    for record in reversed(result):
        # Flip value if this edge's player differs from the perspective we're tracking
        if record['player'] != current_player:
            v = -v
            current_player = record['player']

        record['N'] += 1
        record['W'] += v
        record['Q']  = record['W'] / record['N']

    return result