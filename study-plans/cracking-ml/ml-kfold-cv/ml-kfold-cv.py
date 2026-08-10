import numpy as np

def kfold_cv(X, y, model_fn, k=5, seed=42):
    """
    Returns:
        tuple of (per-fold accuracies, mean accuracy)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)

    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= number of samples")

    rng = np.random.RandomState(seed)

    indices = np.arange(n)
    rng.shuffle(indices)

    folds = np.array_split(indices, k)
    accuracies = []

    for i in range(k):
        validation_indices = folds[i]

        training_indices = np.concatenate([
            folds[j] for j in range(k) if j != i
        ])

        X_train = X[training_indices]
        y_train = y[training_indices]
        X_val = X[validation_indices]
        y_val = y[validation_indices]

        # model_fn returns a prediction function
        predict_fn = model_fn(X_train, y_train)
        predictions = np.asarray(predict_fn(X_val))

        accuracy = np.mean(predictions == y_val)
        accuracies.append(float(accuracy))

    accuracies = np.round(accuracies, 4).tolist()
    mean_accuracy = round(float(np.mean(accuracies)), 4)

    return accuracies, mean_accuracy