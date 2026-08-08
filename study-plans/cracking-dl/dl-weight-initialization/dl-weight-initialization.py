import numpy as np

def weight_init_params(layer_dims, method):
    """
    Returns: list of dicts with keys 'fan_in', 'fan_out', 'shape', 'scale'
    """
    results = []

    for i in range(len(layer_dims) - 1):
        fan_in = layer_dims[i]
        fan_out = layer_dims[i + 1]
        shape = [fan_out, fan_in]

        if method == "random_normal":
            scale = 1.0
        elif method == "xavier_normal":
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        elif method == "xavier_uniform":
            scale = np.sqrt(6.0 / (fan_in + fan_out))
        elif method == "kaiming_normal":
            scale = np.sqrt(2.0 / fan_in)
        elif method == "kaiming_uniform":
            scale = np.sqrt(6.0 / fan_in)
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            "fan_in": fan_in,
            "fan_out": fan_out,
            "shape": shape,
            "scale": round(float(scale), 4)
        })

    return results