import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, dtype=float)

    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nearest_idx = np.argsort(distances)[:k]  # default sort, not kind='stable'
        nearest_labels = y_train[nearest_idx]

        counts = {}
        for label in nearest_labels:
            counts[int(label)] = counts.get(int(label), 0) + 1

        max_count = max(counts.values())
        best_label = min(lbl for lbl, cnt in counts.items() if cnt == max_count)

        predictions.append(best_label)

    return predictions