import numpy as np
from collections import deque

def legal_go_action_mask(board, player, seen_positions, consecutive_passes, move_count):
    """
    Returns: a Boolean vector containing one legality flag per board action and pass.
    """
    board = np.array(board)
    rows, cols = board.shape
    N = rows * cols

    # Terminal check
    is_terminal = (consecutive_passes >= 2) or (move_count >= 2 * N)
    if is_terminal:
        return np.zeros(N + 1, dtype=bool)

    # Convert seen positions to a set of hashable tuples for superko check
    seen_set = {tuple(np.array(p).flatten()) for p in seen_positions}

    def get_group_and_liberties(b, r, c, color):
        """BFS: return (group cells, liberty count) for stone at (r,c)."""
        group = []
        liberties = set()
        visited = set()
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            nr, nc = queue.popleft()
            group.append((nr, nc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                er, ec = nr + dr, nc + dc
                if 0 <= er < rows and 0 <= ec < cols and (er, ec) not in visited:
                    visited.add((er, ec))
                    if b[er, ec] == color:
                        queue.append((er, ec))
                    elif b[er, ec] == 0:
                        liberties.add((er, ec))
        return group, liberties

    def is_legal(r, c):
        # Must be empty
        if board[r, c] != 0:
            return False

        # Work on a copy
        b = board.copy()
        b[r, c] = player
        opponent = -player

        # Remove captured opponent groups (groups with no liberties after placement)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and b[nr, nc] == opponent:
                grp, libs = get_group_and_liberties(b, nr, nc, opponent)
                if len(libs) == 0:
                    for gr, gc in grp:
                        b[gr, gc] = 0

        # Check suicide: played group must have liberties after captures
        _, libs = get_group_and_liberties(b, r, c, player)
        if len(libs) == 0:
            return False

        # Superko: resulting position must not have been seen before
        pos_key = tuple(b.flatten())
        if pos_key in seen_set:
            return False

        return True

    mask = np.zeros(N + 1, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            mask[r * cols + c] = is_legal(r, c)

    # Pass is always legal in nonterminal state
    mask[N] = True

    return mask