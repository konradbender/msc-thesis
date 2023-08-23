from glauber.glauberFixIndices import GlauberSimulatorFixIndices
from glauber.glauberDynIndices import GlauberSimDynIndices
from glauber.glauberTorus import GlauberFixedIndexTorus, GlauberDynIndexTorus
from glauber.glauberSim import GlauberSim
import numpy as np
import logging
import timeit
import pytest
import sys
import os
import shutil


# @pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices, GlauberSimDynIndices))
@pytest.mark.parametrize(
    "class_to_test", (GlauberSimDynIndices, GlauberSimulatorFixIndices)
)
class TestSingleGlauber:
    def test_small_2(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        sim = class_to_test(
            n_interior=84, padding=3, p=0.7, t=2000, tol=0.85, results_dir=tmpdir,
            random_seed=0
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    def test_small_checkpoints(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        

        sim = class_to_test(
            n_interior=84,
            padding=3,
            p=0.7,
            t=1000,
            tol=0.85,
            results_dir=tmpdir,
            random_seed=0
        )
        result_1 = sim.run_single_glauber(False)
        checkpoint_dir = sim.results_dir

        sim_2 = class_to_test(
            n_interior=84,
            padding=3,
            p=0.7,
            t=2000,
            tol=0.85,
            checkpoint_file=f"{checkpoint_dir}/bitmap_results/iter-1000.bmp",
            results_dir=tmpdir + '/continuation',
            random_seed=1
        )
        result_2 = sim_2.run_single_glauber(False)

        assert result_2["fixation"] == False and result_2["iterations"] == 2000

    def test_seed(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=3,
            p=0.7,
            t=2000,
            tol=0.85,
            random_seed=69,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    def test_random_boundary(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=3,
            p=0.7,
            t=2000,
            tol=0.85,
            random_seed=69,
            results_dir=tmpdir,
            boundary="random",
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    def test_zero_padding(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=0,
            p=0.7,
            t=2000,
            tol=0.85,
            random_seed=69,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    @pytest.mark.slow
    def test_large_long(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        sim = class_to_test(
            n_interior=1000,
            padding=100,
            p=0.98,
            t=200000,
            tol=0.982,
            results_dir=tmpdir,
            random_seed=0
        )
        result = sim.run_single_glauber(verbose=True)

        assert result["fixation"] == True
        assert pytest.approx(result["iterations"], 200) == 162854

    @pytest.mark.slow
    def test_small_long(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        sim = class_to_test(
            n_interior=200, padding=15, p=0.7, t=200000, tol=0.9, results_dir=tmpdir,
            random_seed=0
        )
        result = sim.run_single_glauber(verbose=True)
        assert result["fixation"] == True
        assert pytest.approx(result["iterations"], 200) == 196555



@pytest.mark.parametrize("class_to_test", (GlauberSimulatorFixIndices,))
class TestFixedIndices:
    def test_small_1(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            padding=1,
            n_interior=5,
            p=0.7,
            t=200,
            tol=0.98,
            random_seed=0,
            results_dir=tmpdir,
        )
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

    def test_small_bitmap(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=3,
            p=0.95,
            t=2000,
            tol=0.96,
            save_bitmaps_every=500,
            random_seed=0,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000


@pytest.mark.parametrize("class_to_test", (GlauberSimDynIndices,))
class TestDynamicIndices:
    def test_small_1(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            padding=1,
            n_interior=5,
            p=0.7,
            t=200,
            tol=0.98,
            random_seed=0,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(True)
        np.testing.assert_array_equal(
            result["vector"],
            np.array(
                [0.72, 0.76, 0.72, 0.76, 0.72, 0.72, 0.76, 0.72, 0.68, 0.72, 0.76, 0.72, 0.76, 0.76, 0.76, 0.8, 0.8, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.88, 0.88, 0.92, 0.96, 0.96, 0.96, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                ),
        )

    def test_small_bitmap(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=3,
            p=0.95,
            t=2000,
            tol=0.96,
            save_bitmaps_every=500,
            random_seed=0,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == True and result["iterations"] == 249

@pytest.mark.parametrize(
    "class_to_test", (GlauberFixedIndexTorus, GlauberDynIndexTorus)
)
class TestTorus:
    def test_small_2(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        sim = class_to_test(
            n_interior=84, padding=0, p=0.7, t=2000, tol=0.85, results_dir=tmpdir,
            random_seed=0, boundary="random"
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    def test_small_checkpoints(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        

        sim = class_to_test(
            n_interior=84,
            padding=0,
            p=0.7,
            t=1000,
            tol=0.85,
            results_dir=tmpdir,
            random_seed=0,
            boundary="random"
        )
        result_1 = sim.run_single_glauber(False)
        checkpoint_dir = sim.results_dir

        sim_2 = class_to_test(
            n_interior=84,
            padding=0,
            boundary="random",
            p=0.7,
            t=2000,
            tol=0.85,
            checkpoint_file=f"{checkpoint_dir}/bitmap_results/iter-1000.bmp",
            results_dir=tmpdir + '/continuation',
            random_seed=1
        )
        result_2 = sim_2.run_single_glauber(False)

        assert result_2["fixation"] == False and result_2["iterations"] == 2000

    def test_seed(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=0,
            boundary="random",
            p=0.7,
            t=2000,
            tol=0.85,
            random_seed=69,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    def test_random_boundary(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=0,
            p=0.7,
            boundary="random",
            t=2000,
            tol=0.85,
            random_seed=69,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    def test_zero_padding(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        sim = class_to_test(
            n_interior=84,
            padding=0,
            p=0.7,
            t=2000,
            tol=0.85,
            boundary="random",
            random_seed=69,
            results_dir=tmpdir,
        )
        result = sim.run_single_glauber(False)
        assert result["fixation"] == False and result["iterations"] == 2000

    @pytest.mark.slow
    def test_large_long(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        sim = class_to_test(
            n_interior=1000,
            padding=100,
            p=0.98,
            t=200000,
            tol=0.982,
            results_dir=tmpdir,
            random_seed=0
        )
        result = sim.run_single_glauber(verbose=True)

        assert result["fixation"] == True
        assert pytest.approx(result["iterations"], 200) == 162854

    @pytest.mark.slow
    def test_small_long(self, class_to_test: type[GlauberSim], tmpdir):
        tmpdir = str(tmpdir)
        np.random.seed(0)
        sim = class_to_test(
            n_interior=200, padding=15, p=0.7, t=200000, tol=0.9, results_dir=tmpdir,
            random_seed=0
        )
        result = sim.run_single_glauber(verbose=True)
        assert result["fixation"] == True
        assert pytest.approx(result["iterations"], 200) == 196555


if __name__ == "__main__":
    sys.exit(pytest.main(["-c", "pyproject.toml", "--durations=0", 
                          "-k", "test_small_2"]))
