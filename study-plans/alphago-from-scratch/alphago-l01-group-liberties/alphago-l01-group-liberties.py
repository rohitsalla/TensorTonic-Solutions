import numpy as np
from collections import deque

def go_group_liberties(board, row, col):
    """
    Returns: the connected group and its distinct liberties in row-major order.
    """
    board = np.array(board)
    rows, cols = board.shape
    color = board[row, col]

    group     = []
    liberties = []
    visited   = set()
    queue     = deque([(row, col)])
    visited.add((row, col))

    while queue:
        r, c = queue.popleft()
        group.append([r, c])

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                if board[nr, nc] == color:
                    queue.append((nr, nc))
                elif board[nr, nc] == 0:
                    liberties.append([nr, nc])

    return (sorted(group), sorted(liberties))