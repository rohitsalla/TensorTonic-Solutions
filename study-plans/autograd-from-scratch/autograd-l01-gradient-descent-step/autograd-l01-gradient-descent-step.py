import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    theta = np.array(values,    dtype=np.float64)
    g     = np.array(gradients, dtype=np.float64)
    lr    = np.float64(learning_rate)

    # Update: θ' = θ - η·g
    theta_new = theta - lr * g

    # Predicted change: ΔL = Σ g_i · (θ'_i - θ_i) = Σ g_i · (-η·g_i) = -η · ||g||²
    delta_L = float(np.dot(g, theta_new - theta))

    return ([float(v) for v in theta_new], delta_L)