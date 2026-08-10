def topk_accuracy(logits, targets, k):
    """
    Returns: float, mean top-k classification accuracy over N samples, rounded to 4 decimals
    """
    N = len(logits)
    correct = 0

    for i in range(N):
        row = logits[i]
        # stable descending sort by value, ties broken by lower class index
        order = sorted(range(len(row)), key=lambda c: (-row[c], c))
        top_k = order[:k]

        if targets[i] in top_k:
            correct += 1

    accuracy = correct / N
    return round(float(accuracy), 4)