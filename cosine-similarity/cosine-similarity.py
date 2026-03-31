import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    cosine_similarity = dot_product / (norm(a) * norm(b))
    """
    # Write code here
    a_norm,b_norm = np.linalg.norm(a),np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0.0:
        return 0.0
    return np.dot(a,b) / (a_norm * b_norm)