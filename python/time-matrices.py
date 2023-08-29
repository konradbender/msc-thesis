import timeit
import numpy as np
from glauber.DataStructs.BitArrayMat import BitArrayMat 


def run_test(ds, indices):
    """Run test on data struct ds with indices indices"""
    for index in indices:
        ds[index] = 1 - ds[index]
        
if __name__ == '__main__':
    SHAPE = 1000
    P = 0.505
    T = int(1e6)
    
    matrix = np.random.binomial(n=1, p=P, size=SHAPE**2).astype(np.bool_)
    matrix = matrix.reshape((SHAPE, SHAPE))
    bamat = BitArrayMat(SHAPE, SHAPE, matrix.flatten().tolist(), 
                                  wraparound_indices=False)
    
    indices = np.random.randint(0, SHAPE, size=2 * T).reshape((T, 2))
    
    print("Averge time for BitArrayMat:")
    x = timeit.timeit(lambda: run_test(bamat, indices), number=10)
    print(x)
    print("Averge time for Numpy Mat:")
    x = timeit.timeit(lambda: run_test(matrix, indices), number=10)
    print(x)