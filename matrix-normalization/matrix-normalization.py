import numpy as np

class Norm:
    def __init__(self,matrix, axis):
        self.axis = axis
        self.matrix = matrix
        
    def l2(self):
        return np.sqrt(np.sum(self.matrix ** 2, axis=self.axis,keepdims=True))
        
    def l1(self):
        return np.sum(np.abs(self.matrix), axis=self.axis,keepdims=True)
    def max(self):
        return np.max(np.abs(self.matrix), axis=self.axis,keepdims=True)
    def compute(self,norm_type):
        if norm_type == 'l2':
            norm = self.l2()
        elif norm_type == 'l1':
            norm = self.l1()
        elif norm_type == 'max':
            norm = self.max()
        else:
            return None
        norm[norm == 0] = 1
        return self.matrix / norm
            

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        return None
    if axis not in [1,0,None]:
        return None
    norm = Norm(matrix,axis)
    return norm.compute(norm_type) 