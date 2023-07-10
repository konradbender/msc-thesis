import glauber2D
import numpy as np
import logging
import timeit
from numba import njit


def test_single_1():
    np.random.seed(0)
    result = glauber2D.run_single_glauber(np.int64(100), 
                                          np.int64(85), 
                                          np.float64(0.7), 
                                          np.int64(200), 
                                          np.float64(0.85), True)
    
def test_single_2():
    np.random.seed(0)
    result = glauber2D.run_single_glauber(1000, 800, 0.3, 1000, 1, True)
    assert((result["fixation"] == False) and result["iterations"] == 1000)


def test_simulation():
    np.random.seed(0)
    result = glauber2D.run_fixation_simulation(1000, 980, 0.7, 20000, 5, 0.8, True)
    print(result)
    assert(result["fixation_rate"] == 1.0)

if __name__ == "__main__":
    N = 20
    print(timeit.timeit('test_single_1()', number=N, globals=globals())/N)
    print(timeit.timeit('test_single_2()', number=N, globals=globals())/N)
    print(timeit.timeit('test_single_2()', number=N, globals=globals())/N)