import math

def distance_metric(x, y, metric, p=2):
    differences = [
        abs(xi - yi)
        for xi, yi in zip(x, y)
    ]

    if metric == "euclidean":
        distance = math.sqrt(
            sum(diff**2 for diff in differences)
        )

    elif metric == "manhattan":
        distance = sum(differences)

    elif metric == "cosine":
        dot_product = sum(
            xi * yi for xi, yi in zip(x, y)
        )
        norm_x = math.sqrt(sum(xi**2 for xi in x))
        norm_y = math.sqrt(sum(yi**2 for yi in y))

        if norm_x == 0 or norm_y == 0:
            return 0.0

        similarity = dot_product / (norm_x * norm_y)
        similarity = max(-1.0, min(1.0, similarity))
        distance = 1.0 - similarity

    elif metric == "chebyshev":
        distance = max(differences)

    elif metric == "minkowski":
        distance = sum(
            diff**p for diff in differences
        ) ** (1.0 / p)

    else:
        raise ValueError("Unsupported metric")

    return round(distance, 4)