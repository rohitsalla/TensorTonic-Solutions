import copy

def sample_recent_replay_batch(games, window_size, position_indices):
    """
    Returns: selected replay records and the eligible position count.
    """
    # Keep only the most recent window_size games
    recent_games = games[-window_size:] if window_size < len(games) else games

    # Flatten all positions in game order, then position order within each game
    flat = [record for game in recent_games for record in game]

    eligible_count = len(flat)

    # Sample requested positions — repeated indices give independent copies
    sampled = [copy.deepcopy(flat[i]) for i in position_indices]

    return (sampled, eligible_count)