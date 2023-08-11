import cProfile
import numpy as np


import glauber2DBitset as glauber2D


def test_simulation():
    np.random.seed(0)
    sim = glauber2D.GlauberSimulator(1000, 980, 0.7, 20000, 0.8)
    result = sim.run_single_glauber( False)
    print(result)

test_simulation()