from glauberFixIndices import GlauberSimulatorFixIndices
from glauberDynIndices import GlauberSimDynIndices
import numpy as np
import logging
import timeit
from numba import njit
import pytest

@pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices, GlauberSimDynIndices))
class TestClass:
    def test_single_normal(self, class_to_test):
        np.random.seed(0)
        sim = class_to_test(100, 85, 0.7, 200, 0.85)
        result = sim.run_single_glauber(True)
        
    def test_single_numba(self, class_to_test):
        np.random.seed(0)
        sim = class_to_test(100, 85, 0.7, 200, 0.85)
        result = sim.run_single_glauber(True)

    def test_single_2(self, class_to_test):
        np.random.seed(0)
        sim = class_to_test(1000, 800, 0.3, 1000, 1)
        result = sim.run_single_glauber(True)
        assert((result["fixation"] == False) and result["iterations"] == 1000)

    def test_simulation(self, class_to_test):
        np.random.seed(0)
        sim = class_to_test(1000, 980, 0.7, 2000, 0.8)
        result = sim.run_fixation_simulation(5, False)
        print(result)
        assert(result["fixation_rate"] == 1.0)
