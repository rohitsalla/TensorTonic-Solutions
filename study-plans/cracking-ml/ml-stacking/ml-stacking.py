import numpy as np

def stacking_classify(X_train, y_train, X_test, n_folds=3, seed=42):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    classes = np.unique(y_train)

    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    def gini(y):
        if len(y) == 0:
            return 0.0
        imp = 1.0
        for c in np.unique(y):
            p = np.sum(y == c) / len(y)
            imp -= p * p
        return imp

    def fit_stump(X, y):
        n_, d = X.shape
        best_gain = -1.0
        best_feat = 0
        best_thresh = 0.0
        pg = gini(y)
        for feat in range(d):
            for thresh in np.unique(X[:, feat]):
                lm = X[:, feat] <= thresh
                nl = np.sum(lm)
                nr = n_ - nl
                if nl == 0 or nr == 0:
                    continue
                g = pg - (nl / n_) * gini(y[lm]) - (nr / n_) * gini(y[~lm])
                if g > best_gain:
                    best_gain = g
                    best_feat = feat
                    best_thresh = thresh
        lm = X[:, best_feat] <= best_thresh
        lc, lcnt = np.unique(y[lm], return_counts=True) if np.sum(lm) > 0 else (classes, np.ones(len(classes)))
        rc, rcnt = np.unique(y[~lm], return_counts=True) if np.sum(~lm) > 0 else (classes, np.ones(len(classes)))
        return best_feat, best_thresh, lc[np.argmax(lcnt)], rc[np.argmax(rcnt)]

    def predict_stump(X, feat, thresh, ll, rl):
        preds = np.full(X.shape[0], rl)
        preds[X[:, feat] <= thresh] = ll
        return preds

    def predict_knn(X_tr, y_tr, X_te, k=3):
        preds = []
        for x in X_te:
            dists = np.sum((X_tr - x) ** 2, axis=1)
            idx = np.argsort(dists)[:k]
            votes = y_tr[idx]
            uv, cts = np.unique(votes, return_counts=True)
            preds.append(uv[np.argmax(cts)])
        return np.array(preds)

    meta_train = np.zeros((n, 2))
    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        feat, thresh, ll, rl = fit_stump(X_tr, y_tr)
        meta_train[val_idx, 0] = predict_stump(X_val, feat, thresh, ll, rl)
        meta_train[val_idx, 1] = predict_knn(X_tr, y_tr, X_val)

    feat, thresh, ll, rl = fit_stump(X_train, y_train)
    meta_test = np.column_stack([
        predict_stump(X_test, feat, thresh, ll, rl),
        predict_knn(X_train, y_train, X_test)
    ])

    scores = np.zeros((X_test.shape[0], len(classes)))
    for ci, c in enumerate(classes):
        y_m = (y_train == c).astype(float)
        w = np.zeros(2)
        b = 0.0
        lr = 0.1
        for _ in range(500):
            z = meta_train @ w + b
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            dw = (1.0 / n) * (meta_train.T @ (sig - y_m))
            db = (1.0 / n) * np.sum(sig - y_m)
            w -= lr * dw
            b -= lr * db
        z_test = meta_test @ w + b
        scores[:, ci] = 1.0 / (1.0 + np.exp(-np.clip(z_test, -500, 500)))
    return classes[np.argmax(scores, axis=1)].tolist()