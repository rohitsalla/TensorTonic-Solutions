import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lstm_cell(X, h0, c0, Wf, Wi, Wc, Wo, bf, bi, bc, bo, mode="forward", dh_all=None, dc_last=None):
    """
    Returns: Dict with "hidden_states", "cell_states", and optionally gate gradients "dWf"..."dbo" and "dX".
    """
    X = np.array(X, dtype=float)
    h0 = np.array(h0, dtype=float)
    c0 = np.array(c0, dtype=float)
    Wf = np.array(Wf, dtype=float); Wi = np.array(Wi, dtype=float)
    Wc = np.array(Wc, dtype=float); Wo = np.array(Wo, dtype=float)
    bf = np.array(bf, dtype=float); bi = np.array(bi, dtype=float)
    bc = np.array(bc, dtype=float); bo = np.array(bo, dtype=float)

    T, D = X.shape
    H = h0.shape[0]

    hidden_states = []
    cell_states = []
    cache = []

    h_prev, c_prev = h0, c0
    for t in range(T):
        x_t = X[t]
        z = np.concatenate([h_prev, x_t])
        f_t = _sigmoid(Wf @ z + bf)
        i_t = _sigmoid(Wi @ z + bi)
        c_tilde = np.tanh(Wc @ z + bc)
        o_t = _sigmoid(Wo @ z + bo)
        c_t = f_t * c_prev + i_t * c_tilde
        tanh_c_t = np.tanh(c_t)
        h_t = o_t * tanh_c_t

        cache.append(dict(z=z, f_t=f_t, i_t=i_t, c_tilde=c_tilde, o_t=o_t,
                           c_prev=c_prev, c_t=c_t, tanh_c_t=tanh_c_t))
        hidden_states.append(h_t)
        cell_states.append(c_t)
        h_prev, c_prev = h_t, c_t

    hidden_states_arr = np.array(hidden_states)
    cell_states_arr = np.array(cell_states)
    result = {
        "hidden_states": np.round(hidden_states_arr, 4).tolist(),
        "cell_states": np.round(cell_states_arr, 4).tolist(),
    }

    if mode == "backward":
        dh_all = np.array(dh_all, dtype=float)
        dc_next = np.zeros(H) if dc_last is None else np.array(dc_last, dtype=float)

        dWf = np.zeros_like(Wf); dWi = np.zeros_like(Wi)
        dWc = np.zeros_like(Wc); dWo = np.zeros_like(Wo)
        dbf = np.zeros_like(bf); dbi = np.zeros_like(bi)
        dbc = np.zeros_like(bc); dbo = np.zeros_like(bo)
        dX = np.zeros((T, D))

        dh_next = np.zeros(H)

        for t in reversed(range(T)):
            c = cache[t]
            f_t, i_t, c_tilde, o_t = c['f_t'], c['i_t'], c['c_tilde'], c['o_t']
            c_prev, c_t, tanh_c_t, z = c['c_prev'], c['c_t'], c['tanh_c_t'], c['z']

            dh_total = dh_all[t] + dh_next

            do_t = dh_total * tanh_c_t
            do_pre = do_t * o_t * (1 - o_t)

            # Gradient into c_t: from future c_{t+1} plus from h_t = o_t*tanh(c_t)
            dc_total = dc_next + dh_total * o_t * (1 - tanh_c_t ** 2)

            df_t = dc_total * c_prev
            df_pre = df_t * f_t * (1 - f_t)

            di_t = dc_total * c_tilde
            di_pre = di_t * i_t * (1 - i_t)

            dc_tilde = dc_total * i_t
            dc_tilde_pre = dc_tilde * (1 - c_tilde ** 2)

            dc_prev = dc_total * f_t  # becomes dc_next for step t-1

            dz_total = (Wf.T @ df_pre) + (Wi.T @ di_pre) + (Wc.T @ dc_tilde_pre) + (Wo.T @ do_pre)
            dh_prev = dz_total[:H]
            dx_t = dz_total[H:]

            dWf += np.outer(df_pre, z)
            dWi += np.outer(di_pre, z)
            dWc += np.outer(dc_tilde_pre, z)
            dWo += np.outer(do_pre, z)
            dbf += df_pre
            dbi += di_pre
            dbc += dc_tilde_pre
            dbo += do_pre

            dX[t] = dx_t
            dh_next = dh_prev
            dc_next = dc_prev

        result["dWf"] = np.round(dWf, 4).tolist()
        result["dWi"] = np.round(dWi, 4).tolist()
        result["dWc"] = np.round(dWc, 4).tolist()
        result["dWo"] = np.round(dWo, 4).tolist()
        result["dbf"] = np.round(dbf, 4).tolist()
        result["dbi"] = np.round(dbi, 4).tolist()
        result["dbc"] = np.round(dbc, 4).tolist()
        result["dbo"] = np.round(dbo, 4).tolist()
        result["dX"] = np.round(dX, 4).tolist()

    return result