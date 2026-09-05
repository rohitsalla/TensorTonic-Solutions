import torch

def critical_batch_size(batch_sizes, steps_to_target, reached_mask):
    """
    Returns: dictionary containing examples and the critical batch estimate
    """
    examples         = []
    reached_examples = []
    reached_steps    = []

    for B, S, reached in zip(batch_sizes.tolist(), steps_to_target.tolist(), reached_mask.tolist()):
        if reached:
            E = float(B * S)          # explicit float cast
            examples.append(E)
            reached_examples.append(E)
            reached_steps.append(float(S))
        else:
            examples.append(None)

    min_steps    = min(reached_steps)
    min_examples = min(reached_examples)
    bcrit        = min_examples / min_steps

    return {
        "examples":            examples,
        "min_steps":           min_steps,
        "min_examples":        min_examples,
        "critical_batch_size": bcrit,
    }