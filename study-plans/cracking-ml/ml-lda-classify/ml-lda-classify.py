import numpy as np

def lda_classify(X_train, y_train, X_test):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    classes = np.unique(y_train)
    n, d = X_train.shape
    K = len(classes)

    means = {}
    priors = {}
    for c in classes:
        mask = (y_train == c)
        means[c] = X_train[mask].mean(axis=0)
        priors[c] = np.sum(mask) / n

    Sigma = np.zeros((d, d))
    for c in classes:
        diff = X_train[y_train == c] - means[c]
        Sigma += diff.T @ diff
    Sigma /= (n - K)
    Sigma += 1e-6 * np.eye(d)

    Sigma_inv = np.linalg.inv(Sigma)

    W = np.column_stack([Sigma_inv @ means[c] for c in classes])
    b = np.array([-0.5 * means[c] @ Sigma_inv @ means[c] + np.log(priors[c]) for c in classes])

    scores = X_test @ W + b
    pred_indices = np.argmax(scores, axis=1)
    return classes[pred_indices].tolist()
