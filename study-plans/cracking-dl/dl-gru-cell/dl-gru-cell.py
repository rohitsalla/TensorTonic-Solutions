import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gru_cell(X, h0, Wz, Wr, Wh, bz, br, bh, mode="forward", dh_all=None):
    """
    Returns: Dict with "hidden_states" and optionally "dWz", "dWr", "dWh", "dbz", "dbr", "dbh", "dX".
    """
    X = np.array(X, dtype=float)
    h0 = np.array(h0, dtype=float)
    Wz = np.array(Wz, dtype=float)
    Wr = np.array(Wr, dtype=float)
    Wh = np.array(Wh, dtype=float)
    bz = np.array(bz, dtype=float)
    br = np.array(br, dtype=float)
    bh = np.array(bh, dtype=float)

    T, D = X.shape
    H = h0.shape[0]

    hidden_states = []
    cache = []

    h_prev = h0
    for t in range(T):
        x_t = X[t]
        zcat = np.concatenate([h_prev, x_t])
        z_pre = Wz @ zcat + bz
        z_t = _sigmoid(z_pre)
        r_pre = Wr @ zcat + br
        r_t = _sigmoid(r_pre)
        rh = r_t * h_prev
        cat2 = np.concatenate([rh, x_t])
        h_tilde_pre = Wh @ cat2 + bh
        h_tilde = np.tanh(h_tilde_pre)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        cache.append(dict(h_prev=h_prev, zcat=zcat, z_t=z_t, r_t=r_t,
                           cat2=cat2, h_tilde=h_tilde, h_tilde_pre=h_tilde_pre))
        hidden_states.append(h_t)
        h_prev = h_t

    hidden_states_arr = np.array(hidden_states)
    result = {"hidden_states": np.round(hidden_states_arr, 4).tolist()}

    if mode == "backward":
        dh_all = np.array(dh_all, dtype=float)
        dWz = np.zeros_like(Wz)
        dWr = np.zeros_like(Wr)
        dWh = np.zeros_like(Wh)
        dbz = np.zeros_like(bz)
        dbr = np.zeros_like(br)
        dbh = np.zeros_like(bh)
        dX = np.zeros((T, D))

        dh_next = np.zeros(H)

        for t in reversed(range(T)):
            c = cache[t]
            h_prev, z_t, r_t, h_tilde = c['h_prev'], c['z_t'], c['r_t'], c['h_tilde']

            dh_total = dh_all[t] + dh_next

            # Through the interpolation h_t = (1-z)*h_prev + z*h_tilde
            dz_t = dh_total * (h_tilde - h_prev)
            dz_pre = dz_t * z_t * (1 - z_t)

            dh_tilde = dh_total * z_t
            dh_tilde_pre = dh_tilde * (1 - h_tilde ** 2)

            # Through candidate: cat2 = [r*h_prev, x_t]
            dcat2 = Wh.T @ dh_tilde_pre
            d_rh = dcat2[:H]
            dx_from_h = dcat2[H:]

            d_r_t = d_rh * h_prev
            dh_prev_from_rh = d_rh * r_t
            d_r_pre = d_r_t * r_t * (1 - r_t)

            # Through gates: zcat = [h_prev, x_t]
            dzcat_from_z = Wz.T @ dz_pre
            dzcat_from_r = Wr.T @ d_r_pre

            dh_prev_from_z, dx_from_z = dzcat_from_z[:H], dzcat_from_z[H:]
            dh_prev_from_r, dx_from_r = dzcat_from_r[:H], dzcat_from_r[H:]

            dh_prev_direct = dh_total * (1 - z_t)

            dh_next = dh_prev_direct + dh_prev_from_rh + dh_prev_from_z + dh_prev_from_r
            dX[t] = dx_from_h + dx_from_z + dx_from_r

            dWz += np.outer(dz_pre, c['zcat'])
            dWr += np.outer(d_r_pre, c['zcat'])
            dWh += np.outer(dh_tilde_pre, c['cat2'])
            dbz += dz_pre
            dbr += d_r_pre
            dbh += dh_tilde_pre

        result["dWz"] = np.round(dWz, 4).tolist()
        result["dWr"] = np.round(dWr, 4).tolist()
        result["dWh"] = np.round(dWh, 4).tolist()
        result["dbz"] = np.round(dbz, 4).tolist()
        result["dbr"] = np.round(dbr, 4).tolist()
        result["dbh"] = np.round(dbh, 4).tolist()
        result["dX"] = np.round(dX, 4).tolist()

    return result