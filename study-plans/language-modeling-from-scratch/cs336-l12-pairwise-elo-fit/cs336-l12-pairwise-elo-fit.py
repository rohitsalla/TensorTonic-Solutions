import math
import torch

def pairwise_elo_fit(num_models, model_a, model_b, outcomes,
                     steps, learning_rate):
    """
    Returns: dictionary containing Elo results and graph connectivity
    """
    ratings = torch.zeros(num_models, dtype=torch.float64, device=outcomes.device)
    elo_scale = math.log(10.0) / 400.0
    for _ in range(steps):
        probabilities = torch.sigmoid(
            elo_scale * (ratings[model_a] - ratings[model_b])
        )
        residuals = outcomes.double() - probabilities
        gradient = torch.zeros_like(ratings)
        if residuals.numel() > 0:
            gradient.index_add_(0, model_a, elo_scale * residuals)
            gradient.index_add_(0, model_b, -elo_scale * residuals)
            gradient /= residuals.numel()
        ratings = ratings + learning_rate * gradient
        ratings = ratings - ratings.mean()

    fitted_probabilities = torch.sigmoid(
        elo_scale * (ratings[model_a] - ratings[model_b])
    )
    ranking_values = sorted(
        range(num_models), key=lambda index: (-float(ratings[index]), index)
    )
    rankings = torch.tensor(
        ranking_values, dtype=torch.int64, device=outcomes.device
    )

    parents = list(range(num_models))

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for left, right in zip(model_a.tolist(), model_b.tolist()):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    root_ids = {}
    component_values = []
    for model in range(num_models):
        root = find(model)
        if root not in root_ids:
            root_ids[root] = len(root_ids)
        component_values.append(root_ids[root])
    component_ids = torch.tensor(
        component_values, dtype=torch.int64, device=outcomes.device
    )
    return {
        "ratings": ratings,
        "rankings": rankings,
        "fitted_probabilities": fitted_probabilities,
        "connected": len(root_ids) == 1,
        "component_ids": component_ids,
    }
