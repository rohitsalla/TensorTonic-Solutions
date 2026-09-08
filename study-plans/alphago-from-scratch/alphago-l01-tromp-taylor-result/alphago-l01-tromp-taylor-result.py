import numpy as np
from collections import deque

def tromp_taylor_result(board, komi, consecutive_passes, move_count):
    """
    Returns: terminal status, black score, white score, and the winning colour.
    """
    board = np.array(board)
    rows, cols = board.shape
    N = rows * cols

    # Terminal condition: 2 consecutive passes OR move_count >= 2 * N
    is_terminal = (consecutive_passes >= 2) or (move_count >= 2 * N)
    if not is_terminal:
        return (False, None, None, None)

    # Count stones
    black_stones = int(np.sum(board == 1))
    white_stones = int(np.sum(board == -1))

    # Flood-fill each connected empty region
    black_territory = 0
    white_territory = 0
    visited = np.zeros((rows, cols), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 0 or visited[r, c]:
                continue

            # BFS to find this empty region and its boundary colors
            region = []
            boundary_colors = set()
            queue = deque([(r, c)])
            visited[r, c] = True

            while queue:
                nr, nc = queue.popleft()
                region.append((nr, nc))

                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    er, ec = nr + dr, nc + dc
                    if 0 <= er < rows and 0 <= ec < cols:
                        if board[er, ec] == 0 and not visited[er, ec]:
                            visited[er, ec] = True
                            queue.append((er, ec))
                        elif board[er, ec] != 0:
                            boundary_colors.add(int(board[er, ec]))

            # Award territory only if bordered by exactly one color
            size = len(region)
            if boundary_colors == {1}:
                black_territory += size
            elif boundary_colors == {-1}:
                white_territory += size
            # else: neutral (touches both or neither)

    black_score = np.float64(black_stones + black_territory)
    white_score = np.float64(white_stones + white_territory + komi)

    if black_score > white_score:
        winner = 1
    elif white_score > black_score:
        winner = -1
    else:
        winner = 0

    return (True, black_score, white_score, winner)