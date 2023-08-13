from glauberFixIndices import GlauberSimulatorFixIndices
from glauberDynIndices import GlauberSimDynIndices
from glauberSim import GlauberSim
import numpy as np
import logging
import timeit
import pytest
import sys


# @pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices, GlauberSimDynIndices))
@pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices,))
class TestSingleGlauber:
    def test_small_1(self, class_to_test: type[GlauberSim]):
        np.random.seed(0)
        sim = class_to_test(padding=1, n_interior=5, p=0.7, t=200, tol=0.98)
        result = sim.run_single_glauber(True)
        np.testing.assert_array_equal(
            result["vector"],
            np.array(
                [
                    0.68,
                    0.68,
                    0.68,
                    0.68,
                    0.68,
                    0.68,
                    0.68,
                    0.72,
                    0.68,
                    0.72,
                    0.76,
                    0.8,
                    0.84,
                    0.84,
                    0.84,
                    0.88,
                    0.92,
                    0.92,
                    0.92,
                    0.96,
                    0.96,
                    0.96,
                    0.96,
                    0.96,
                    0.96,
                    0.96,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            ),
        )

    def test_small_2(self, class_to_test: type[GlauberSim]):
        np.random.seed(0)
        sim = class_to_test(n_interior=84, padding=3, p=0.7, t=2000, tol=0.85)
        result = sim.run_single_glauber(False)
        assert(result["fixation"] == False and result["iterations"] == 2000)


    def test_large(self, class_to_test: type[GlauberSim]):
        np.random.seed(0)
        sim = class_to_test(n_interior=2000, padding=100, p=0.98, t=200000, tol=0.99)
        result = sim.run_single_glauber(False)
        assert(result["fixation"] == False and result["iterations"] == 2000)



    def test_single_2(self, class_to_test):
        np.random.seed(0)
        sim = class_to_test(1000, 800, 0.3, 1000, 1)
        result = sim.run_single_glauber(True)
        assert (result["fixation"] == False) and result["iterations"] == 1000


# @pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices, GlauberSimDynIndices))
@pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices,))
class TestFixationSimulation:
    def test_simulation(self, class_to_test):
        np.random.seed(0)
        sim = class_to_test(1000, 980, 0.7, 2000, 0.8)
        result = sim.run_fixation_simulation(5, False)
        print(result)
        assert result["fixation_rate"] == 1.0


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [
                "-c",
                "/Users/konrad/code/school/msc-thesis/python/pyproject.toml",
                "-k test_large",
            ]
        )
    )
