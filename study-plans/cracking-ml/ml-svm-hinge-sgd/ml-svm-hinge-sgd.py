import numpy as np

def svm_sgd(X_train, y_train, X_test, lr=0.01, lam=0.01, n_epochs=100):
    """
    Train an SVM using SGD on hinge loss with L2 regularization.
    Parameters:
    - X_train: Training feature matrix (n samples, d features)
    - y_train: Training labels (-1 or +1)
    - X_test: Test feature matrix
    - lr: Learning rate
    - lam: L2 regularization strength
    - n_epochs: Number of training epochs
    Returns: list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    n, d = X_train.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(n_epochs):
        for i in range(n):
            xi = X_train[i]
            yi = y_train[i]
            margin = yi * (np.dot(w, xi) + b)
            if margin < 1:
                w = w - lr * (lam * w - yi * xi)
                b = b + lr * yi
            else:
                w = w - lr * lam * w

    predictions = []
    for x in X_test:
        score = np.dot(w, x) + b
        predictions.append(1 if score > 0 else -1)

    return predictions