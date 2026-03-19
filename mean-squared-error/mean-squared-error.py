import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred_Array = np.asarray(y_pred,dtype=np.float64)
    y_true_Array = np.asarray(y_true, dtype=np.float64)
    if y_pred_Array.shape != y_true_Array.shape:
        return None
    squared = (y_pred_Array - y_true_Array) ** 2
    results = np.sum(squared) / len(y_pred)
    return results