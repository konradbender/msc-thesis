import glauber2D
import numpy as np
import logging
import timeit
from numba import njit, jit


def time_single_normal():
    np.random.seed(0)
    result = glauber2D.run_single_glauber.py_func(np.int64(1000), 
                                                  np.int64(850), 
                                                  np.float64(0.7), 
                                                  np.int64(200000), 
                                                  np.float64(0.85), verbose=False)
@jit(cache=True, debug=2, nopython=True)    
def time_single_numba():
    np.random.seed(0)
    result = glauber2D.run_single_glauber(np.int64(1000), 
                                          np.int64(850), 
                                          np.float64(0.7), 
                                          np.int64(200000), 
                                          np.float64(0.85), verbose=False)


if __name__ == "__main__":
    N = 3
    logging.basicConfig(level=logging.WARN)
    print("Runtime normal:", timeit.timeit('time_single_normal()', number=N, globals=globals())/N)
    print("Runtime numba:", timeit.timeit('time_single_numba()', number=N, globals=globals())/N)

