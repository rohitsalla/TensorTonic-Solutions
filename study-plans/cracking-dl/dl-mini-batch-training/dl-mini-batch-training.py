import numpy as np

def _forward(x, weights, biases):
    a = np.array(x, dtype=float)
    activations = [a]
    zs = []
    num_layers = len(weights)
    for l in range(num_layers):
        z = weights[l] @ a + biases[l]
        zs.append(z)
        a = np.maximum(0, z) if l < num_layers - 1 else z
        activations.append(a)
    return activations, zs

def _backward(activations, zs, y, weights):
    num_layers = len(weights)
    grads_W = [None] * num_layers
    grads_b = [None] * num_layers
    y = np.array(y, dtype=float)

    # Output layer is linear, loss is 1/2||a - y||^2 -> delta = a - y
    delta = activations[-1] - y

    for l in reversed(range(num_layers)):
        a_prev = activations[l]
        grads_W[l] = np.outer(delta, a_prev)
        grads_b[l] = delta
        if l > 0:
            da_prev = weights[l].T @ delta
            relu_deriv = (zs[l - 1] > 0).astype(float)
            delta = da_prev * relu_deriv

    return grads_W, grads_b

def _full_loss(X, y, weights, biases):
    n = len(X)
    total = 0.0
    for i in range(n):
        activations, _ = _forward(X[i], weights, biases)
        yi = np.array(y[i], dtype=float)
        total += np.sum((activations[-1] - yi) ** 2)
    return total / (2 * n)

def mini_batch_training(X, y, weights, biases, lr, epochs, batch_size):
    """
    Returns: list of floats
    """
    weights = [np.array(w, dtype=float) for w in weights]
    biases = [np.array(b, dtype=float) for b in biases]
    n = len(X)
    losses = []

    for _ in range(epochs):
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            bsize = end - start

            sum_grads_W = [np.zeros_like(w) for w in weights]
            sum_grads_b = [np.zeros_like(b) for b in biases]

            for i in range(start, end):
                activations, zs = _forward(X[i], weights, biases)
                gW, gb = _backward(activations, zs, y[i], weights)
                for l in range(len(weights)):
                    sum_grads_W[l] += gW[l]
                    sum_grads_b[l] += gb[l]

            for l in range(len(weights)):
                weights[l] -= lr * sum_grads_W[l] / bsize
                biases[l] -= lr * sum_grads_b[l] / bsize

            start = end

        losses.append(round(float(_full_loss(X, y, weights, biases)), 4))

    return losses