import torch

def encode_go_history(history, to_play):
    """
    Returns: a floating 17-plane current-player-relative history tensor.
    """
    # Get board shape and device from the first (most recent) history entry
    board0 = torch.as_tensor(history[0], dtype=torch.float32)
    rows, cols = board0.shape
    device = board0.device

    planes = []

    # 8 history steps, newest first; missing steps are zero-padded
    for t in range(8):
        if t < len(history):
            b = torch.as_tensor(history[t], dtype=torch.float32).to(device)
            current_plane  = (b == to_play).float()    # 1 where current player has stone
            opponent_plane = (b == -to_play).float()   # 1 where opponent has stone
        else:
            current_plane  = torch.zeros(rows, cols, dtype=torch.float32, device=device)
            opponent_plane = torch.zeros(rows, cols, dtype=torch.float32, device=device)

        planes.append(current_plane)
        planes.append(opponent_plane)

    # Final plane: all ones if Black (1) to play, all zeros if White (-1) to play
    color_plane = torch.full((rows, cols), 1.0 if to_play == 1 else 0.0,
                             dtype=torch.float32, device=device)
    planes.append(color_plane)

    return torch.stack(planes, dim=0)   # (17, rows, cols)