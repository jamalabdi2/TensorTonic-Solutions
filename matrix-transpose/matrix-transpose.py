import numpy as np

def matrix_transpose(A):
    A = np.asarray(A)
    rows,cols = A.shape
    new_arr = np.zeros((cols,rows), dtype=A.dtype)

    for i in range(rows):
        for j in range(cols):
            new_arr[j,i] = A[i,j]
    return new_arr
    
