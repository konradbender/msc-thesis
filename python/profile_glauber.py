import cProfile
import numpy as np

import glauber2DBitset as glauber2D

def test_simulation():
    np.random.seed(0)
    result = glauber2D.run_single_glauber(1000, 980, 0.7, 20000, 0.8, False)
    print(result)

cProfile.run('test_simulation()')