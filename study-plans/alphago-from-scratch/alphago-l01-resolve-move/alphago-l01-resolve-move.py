import numpy as np
from collections import deque

def resolve_go_action(board, player, action, consecutive_passes, move_count):
    """
    Returns: the next board, player, captured count, pass count, move count, and terminal flag.
    """
    board = np.array(board)
    rows, cols = board.shape
    N = rows * cols

    def get_group_and_liberties(b, r, c, color):
        group = []; liberties = set(); visited = set()
        queue = deque([(r, c)]); visited.add((r, c))
        while queue:
            nr, nc = queue.popleft(); group.append((nr, nc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                er, ec = nr+dr, nc+dc
                if 0 <= er < rows and 0 <= ec < cols and (er,ec) not in visited:
                    visited.add((er, ec))
                    if b[er, ec] == color: queue.append((er, ec))
                    elif b[er, ec] == 0:   liberties.add((er, ec))
        return group, liberties

    move_count += 1

    if action == N:
        # Pass: board unchanged, consecutive passes increment
        consecutive_passes += 1
        captured = 0
        next_board = board.copy()
    else:
        # Intersection action
        r, c = action // cols, action % cols
        consecutive_passes = 0
        next_board = board.copy()
        next_board[r, c] = player
        opponent = -player

        # Remove captured opponent groups (no liberties after placement)
        captured = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and next_board[nr, nc] == opponent:
                grp, libs = get_group_and_liberties(next_board, nr, nc, opponent)
                if len(libs) == 0:
                    for gr, gc in grp:
                        next_board[gr, gc] = 0
                    captured += len(grp)

    next_player = -player
    is_terminal = (consecutive_passes >= 2) or (move_count >= 2 * N)

    return (next_board.tolist(), next_player, captured, consecutive_passes, move_count, is_terminal)