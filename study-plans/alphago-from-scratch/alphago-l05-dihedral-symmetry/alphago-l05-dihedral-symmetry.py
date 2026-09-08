import numpy as np

def apply_go_symmetry(state_planes, policy, value, transform_id):
    """
    Returns: transformed state planes, transformed policy, and unchanged value.
    """
    state  = np.array(state_planes)   # (planes, N, N)
    pol    = np.array(policy)         # (N*N + 1,)
    N      = state.shape[1]

    # Separate board moves from pass
    board_pol = pol[:-1].reshape(N, N)   # (N, N)
    pass_prob = pol[-1]

    def transform_2d(arr):
        """Apply transform to a 2D (N, N) array."""
        t = transform_id
        if t >= 4:
            arr = np.fliplr(arr)   # reflect left-right first
            t  -= 4
        # Rotate counterclockwise by t * 90 degrees
        return np.rot90(arr, k=t)

    # Transform each state plane (last two axes)
    new_state = np.stack([transform_2d(state[p]) for p in range(state.shape[0])])

    # Transform board policy the same way
    new_board_pol = transform_2d(board_pol)
    new_policy    = np.append(new_board_pol.flatten(), pass_prob)

    return (new_state, new_policy, value)