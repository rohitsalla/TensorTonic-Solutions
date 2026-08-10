import numpy as np

def precision_recall_ap(y_true, y_scores):
    """
    Returns a tuple of:
        (recall_list, precision_list, ap_value)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    total_positives = np.sum(y_true == 1)

    if total_positives == 0:
        raise ValueError("y_true must contain at least one positive label")

    # Sort by descending score
    order = np.argsort(-y_scores, kind="mergesort")
    sorted_scores = y_scores[order]
    sorted_labels = y_true[order]

    # PR curve starts at recall=0, precision=1
    recall = [0.0]
    precision = [1.0]

    true_positives = 0
    false_positives = 0
    average_precision = 0.0
    previous_recall = 0.0

    i = 0
    n = len(y_true)

    # Process samples with equal scores as one threshold
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

        current_recall = true_positives / total_positives
        current_precision = (
            true_positives
            / (true_positives + false_positives)
        )

        recall.append(current_recall)
        precision.append(current_precision)

        # Step-function integration
        average_precision += (
            current_recall - previous_recall
        ) * current_precision

        previous_recall = current_recall
        i = j

    recall = np.round(recall, 4).tolist()
    precision = np.round(precision, 4).tolist()
    average_precision = round(float(average_precision), 4)

    return recall, precision, average_precision