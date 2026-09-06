import numpy as np

def value_multiplication_node(left, right, output_id):
    """
    Returns: a multiplication node that retains the two supplied leaf records as ordered parents
    """
    return {
        'id':      output_id,
        'data':    float(np.float64(left['data']) * np.float64(right['data'])),
        'grad':    0.0,
        'op':      '*',
        'parents': [left, right],
    }