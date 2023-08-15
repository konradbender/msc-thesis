from glauberFixIndices import GlauberSimulatorFixIndices
from glauberDynIndices import GlauberSimDynIndices
from glauberSim import GlauberSim
import numpy as np
import logging
import timeit
import pytest
import sys
import os


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

    @pytest.mark.slow
    def test_large_long(self, class_to_test: type[GlauberSim]):
        np.random.seed(0)
        sim = class_to_test(n_interior=1000, padding=100, p=0.98, t=200000, tol=0.982)
        result = sim.run_single_glauber(verbose=True)

        assert result["fixation"] == True
        assert pytest.approx(result["iterations"], 200) == 162854

    @pytest.mark.slow
    def test_small_long(self, class_to_test: type[GlauberSim]):
        np.random.seed(0)
        sim = class_to_test(n_interior=200, padding=15, p=0.7, t=200000, tol=0.9)
        result = sim.run_single_glauber(verbose=True)
        assert result["fixation"] == True
        assert pytest.approx(result["iterations"], 200) == 196555

    def test_small_bitmap(self, class_to_test: type[GlauberSim]):
        np.random.seed(0)
        sim = class_to_test(n_interior=84, padding=3, p=0.95, t=2000, tol=0.96,
                            save_bitmaps_every=500)
        result = sim.run_single_glauber(False)
        assert(result["fixation"] == False and result["iterations"] == 2000)
    


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
                "pyproject.toml",
                "-k not simulation",
                "--durations=0"
            ]
        )
    )
