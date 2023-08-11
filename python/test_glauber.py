import glauber2DBitset as glauber2D
import numpy as np
import logging
import timeit
from numba import njit


def test_single_normal():
    np.random.seed(0)
    sim = glauber2D.GlauberSimulator(100, 85, 0.7, 200, 0.85)
    result = sim.run_single_glauber(True)
    
def test_single_numba():
    np.random.seed(0)
    sim = glauber2D.GlauberSimulator(100, 85, 0.7, 200, 0.85)
    result = sim.run_single_glauber(True)

def test_single_2():
    np.random.seed(0)
    sim = glauber2D.GlauberSimulator(1000, 800, 0.3, 1000, 1)
    result = sim.run_single_glauber(True)
    assert((result["fixation"] == False) and result["iterations"] == 1000)


def test_simulation():
    np.random.seed(0)
    sim = glauber2D.GlauberSimulator(1000, 980, 0.7, 20000, 0.8)
    result = sim.run_fixation_simulation(5, False)
    print(result)
    assert(result["fixation_rate"] == 1.0)
