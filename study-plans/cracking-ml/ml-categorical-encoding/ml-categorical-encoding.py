import numpy as np

def categorical_encode(data, method="label"):
    """
    Returns the encoded result based on the selected method.
    """
    if method not in ("label", "onehot"):
        raise ValueError("method must be either 'label' or 'onehot'")

    classes = sorted(set(data))
    category_to_index = {
        category: index
        for index, category in enumerate(classes)
    }

    encoded = [
        category_to_index[category]
        for category in data
    ]

    if method == "label":
        return {
            "encoded": encoded,
            "classes": classes
        }

    onehot = np.zeros(
        (len(data), len(classes)),
        dtype=int
    )

    onehot[np.arange(len(data)), encoded] = 1

    return onehot.tolist()