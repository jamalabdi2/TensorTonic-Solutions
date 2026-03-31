import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    N = len(A)
    trace = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                trace += A[i][j]
    return trace
