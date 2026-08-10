import numpy as np

def auc_roc(y_true, y_scores):
    """
    Returns a tuple of (fpr_list, tpr_list, auc_value).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        raise ValueError(
            "y_true must contain at least one positive and one negative"
        )

    # Sort samples by descending score
    order = np.argsort(-y_scores, kind="mergesort")
    sorted_scores = y_scores[order]
    sorted_labels = y_true[order]

    # ROC curve starts at (0, 0)
    fpr = [0.0]
    tpr = [0.0]

    true_positives = 0
    false_positives = 0
    i = 0
    n = len(y_true)

    # Process equal scores together as one threshold
    while i < n:
        j = i

        while (
            j < n
            and sorted_scores[j] == sorted_scores[i]
        ):
            if sorted_labels[j] == 1:
                true_positives += 1
            else:
                false_positives += 1

            j += 1

        tpr.append(true_positives / positives)
        fpr.append(false_positives / negatives)
        i = j

    # Trapezoidal integration
    auc = 0.0

    for i in range(1, len(fpr)):
        width = fpr[i] - fpr[i - 1]
        average_height = (tpr[i] + tpr[i - 1]) / 2.0
        auc += width * average_height

    fpr = np.round(fpr, 4).tolist()
    tpr = np.round(tpr, 4).tolist()
    auc = round(float(auc), 4)

    return fpr, tpr, auc