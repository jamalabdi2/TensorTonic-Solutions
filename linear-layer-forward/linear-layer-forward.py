def get_rows_column(matrix):
    return len(matrix), len(matrix[0])

def generate_result(m,p):
    return [[0 for _ in range(p)] for _ in range(m)]

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    NAIVE IMPLEMENTATION three nested loop, not pythonic
    """
    # Write code here
    m,n = get_rows_column(X)
    _,p = get_rows_column(W)
    #I should check if inner dimensions does not work and raise a shape mismatch error, ignore it for now
    result = generate_result(m,p)
    for i in range(m):
        for j in range(p):
            dot_product = 0
            for k in range(n):
                dot_product += X[i][k] * W[k][j]
            result[i][j] = dot_product + b[j]
    return result
