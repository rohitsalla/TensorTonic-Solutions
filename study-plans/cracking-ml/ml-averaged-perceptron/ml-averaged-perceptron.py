import numpy as np

def averaged_perceptron(X_train, y_train, X_test, n_epochs=10):
    """
    Returns: A list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    n, d = X_train.shape

    w = np.zeros(d)
    b = 0.0

    w_sum = np.zeros(d)
    b_sum = 0.0
    count = 0

    for _ in range(n_epochs):
        for i in range(n):
            xi = X_train[i]
            yi = y_train[i]
            if yi * (np.dot(w, xi) + b) <= 0:
                w = w + yi * xi
                b = b + yi
            w_sum += w
            b_sum += b
            count += 1

    w_avg = w_sum / count
    b_avg = b_sum / count

    predictions = []
    for x in X_test:
        score = np.dot(w_avg, x) + b_avg
        predictions.append(1 if score > 0 else -1)

    return predictions