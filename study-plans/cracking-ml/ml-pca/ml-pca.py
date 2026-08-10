import numpy as np

def pca(X, n_components=2):
    """
    Returns a tuple of:
        (transformed_data, explained_variance_ratios)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if not 1 <= n_components <= d:
        raise ValueError(
            "n_components must be between 1 and the number of features"
        )

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Unbiased covariance matrix
    covariance = (X_centered.T @ X_centered) / (n - 1)

    # Eigenvalues are returned in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort in descending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Select the top components without changing their signs
    components = eigenvectors[:, :n_components]

    # Project onto principal components
    transformed = X_centered @ components

    # Remove tiny negative eigenvalues from numerical error
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = np.sum(eigenvalues)

    if total_variance == 0:
        ratios = np.zeros(n_components)
    else:
        ratios = eigenvalues[:n_components] / total_variance

    transformed = np.round(transformed, 4)
    ratios = np.round(ratios, 4)

    transformed[transformed == 0] = 0.0
    ratios[ratios == 0] = 0.0

    return transformed.tolist(), ratios.tolist()