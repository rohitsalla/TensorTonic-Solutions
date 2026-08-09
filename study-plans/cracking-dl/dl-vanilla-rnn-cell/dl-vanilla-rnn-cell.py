import numpy as np

def rnn_cell(X, h0, Wxh, Whh, bh, mode="forward", dh_all=None):
    """
    Returns: Dict with "hidden_states" and optionally "dWxh", "dWhh", "dbh", "dX".
    """
    X = np.array(X, dtype=float)
    h0 = np.array(h0, dtype=float)
    Wxh = np.array(Wxh, dtype=float)
    Whh = np.array(Whh, dtype=float)
    bh = np.array(bh, dtype=float)

    T, D = X.shape
    H = h0.shape[0]

    hidden_states = []
    pre_acts = []   # z_t = Whh@h_{t-1} + Wxh@x_t + bh, needed for tanh derivative
    prevs = []      # h_{t-1} used at each step, needed for dWhh

    h_prev = h0
    for t in range(T):
        z = Whh @ h_prev + Wxh @ X[t] + bh
        h_t = np.tanh(z)
        pre_acts.append(z)
        prevs.append(h_prev)
        hidden_states.append(h_t)
        h_prev = h_t

    hidden_states_arr = np.array(hidden_states)
    result = {"hidden_states": np.round(hidden_states_arr, 4).tolist()}

    if mode == "backward":
        dh_all = np.array(dh_all, dtype=float)
        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dbh = np.zeros_like(bh)
        dX = np.zeros((T, D))

        dh_next = np.zeros(H)  # gradient flowing back from h_{t+1}

        for t in reversed(range(T)):
            dh_total = dh_all[t] + dh_next          # upstream + future gradient
            dz = (1 - np.tanh(pre_acts[t]) ** 2) * dh_total

            dWxh += np.outer(dz, X[t])
            dWhh += np.outer(dz, prevs[t])
            dbh += dz
            dX[t] = Wxh.T @ dz
            dh_next = Whh.T @ dz

        result["dWxh"] = np.round(dWxh, 4).tolist()
        result["dWhh"] = np.round(dWhh, 4).tolist()
        result["dbh"] = np.round(dbh, 4).tolist()
        result["dX"] = np.round(dX, 4).tolist()

    return result