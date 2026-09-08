import numpy as np

def neural_guided_mcts(root_state, transitions, legal_actions, players,
                       policy_logits, value_predictions, terminal_values,
                       simulations, cpuct, temperature):
    """
    Returns: root visit counts, root mean values, and the root search policy.
    """
    edge_stats = {}   # state -> {action -> {N, W, Q, P}}

    def softmax(logits):
        l = np.array(logits, dtype=np.float64)
        e = np.exp(l - l.max()); return e / e.sum()

    def expand(state):
        actions = legal_actions[state]
        priors  = softmax(policy_logits[state])
        edge_stats[state] = {
            a: {"N": 0, "W": 0.0, "Q": 0.0, "P": float(priors[i])}
            for i, a in enumerate(actions)
        }

    def puct_select(state):
        stats = edge_stats[state]
        actions = legal_actions[state]   # strictly increasing = lowest index first on tie
        T = sum(stats[a]["N"] for a in actions)
        sqrt_T = np.sqrt(T)
        best_score = -np.inf; best_action = None
        for a in actions:
            score = stats[a]["Q"] + cpuct * stats[a]["P"] * sqrt_T / (1 + stats[a]["N"])
            if score > best_score:   # strict > preserves first (lowest) on tie
                best_score = score; best_action = a
        return best_action

    # Expand root before any simulation
    expand(root_state)

    for _ in range(simulations):
        state = root_state; path = []

        # ── Selection: descend until terminal or unexpanded ───────────────
        while True:
            if state in terminal_values:
                leaf_val = float(terminal_values[state])
                state_player = players[state]; break
            if state not in edge_stats:
                expand(state)
                leaf_val = float(value_predictions[state])
                state_player = players[state]; break
            action = puct_select(state)
            path.append((state, action))
            state = transitions[state][action]

        # ── Backup: sign-flip whenever player perspective changes ─────────
        v = leaf_val; current_player = state_player
        for st, ac in reversed(path):
            edge_player = players[st]
            if edge_player != current_player:
                v = -v; current_player = edge_player
            s = edge_stats[st][ac]
            s["N"] += 1; s["W"] += v; s["Q"] = s["W"] / s["N"]

    # ── Extract root statistics in legal-action order ─────────────────────
    root_actions = legal_actions[root_state]
    visits    = np.array([edge_stats[root_state][a]["N"] for a in root_actions], dtype=np.int64)
    mean_vals = np.array([edge_stats[root_state][a]["Q"] for a in root_actions], dtype=np.float64)

    # Temperature-adjusted policy
    if temperature == 0:
        policy = np.zeros(len(root_actions), dtype=np.float64)
        policy[int(np.argmax(visits))] = 1.0   # lowest index wins on tie via argmax
    else:
        powered = visits.astype(np.float64) ** (1.0 / temperature)
        policy  = powered / powered.sum() if powered.sum() > 0 else \
                  np.ones(len(root_actions), dtype=np.float64) / len(root_actions)

    return (visits, mean_vals, policy)