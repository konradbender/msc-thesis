from abc import ABC, abstractmethod
import numpy as np
import os
import datetime
import json


class GlauberSim(ABC):
    def __init__(
        self,
        n_interior: np.int64,
        p: np.float64,
        t: np.int64,
        tol: np.float64,
        padding: np.int64 = None,  # for new code
        n_outer: np.int64 = None,  # kept for depreceated compatibility
        save_bitmaps_every=None,
        results_dir=None,
    ) -> None:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1

        Parameters
        ----------
        n_interior : int
            dimension of inner lattice
        p : float
            probability that vertices are -1 at t=0
        t : int
            number of vertex-updates to perform
        tol : float
            minimum share of -1 vertices to reach to declare fixation
        padding : int
            number of vertices beyond the border of n_interior*n_interior matrix
        save_bitmaps_every : int
            number of iterations after which to save a bitmap of the matrix. If None, is never saved
        results_dir : str
            directory to save results in. If None, results are saved in a directory named after the current time
        """
        if padding is None and n_outer is None:
            raise ValueError("padding or n_outer must be specified")

        if padding is not None:
            self.padding = padding
            self.n_outer = n_interior + 2 * padding

        if n_outer is not None:
            raise DeprecationWarning("n_outer is deprecated, use padding instead")
            self.n_outer = self.n_outer
            self.padding = (self.n_outer - n_interior) // 2

        self.n_interior = n_interior
        self.p = p
        self.t = t
        self.tol = tol
        self.save_bitmaps_every = save_bitmaps_every
        self.results_dir = results_dir

        if self.results_dir is None:
            now = datetime.datetime.now()
            time_string = now.strftime("%m%d_%H-%M-%S")
            self.results_dir = f"./results/{time_string}/"

        os.makedirs(self.results_dir, exist_ok=True)

        parameters = {
            "n_interior": self.n_interior,
            "p": self.p,
            "t": self.t,
            "tol": self.tol,
            "padding": self.padding,
            "save_bitmaps_every": self.save_bitmaps_every,
            "results_dir": self.results_dir,
        }
        with open(f"{self.results_dir}/simulation-params.json", "w") as f:
            json.dump(parameters, f)

    @abstractmethod
    def run_single_glauber(*args, **kwargs) -> dict:
        raise NotImplementedError()

    # This is not tested yet
    def run_fixation_simulation(self, iter, verbose: bool = False) -> dict:
        fixations = 0
        iterations = 0
        results = []

        for i in range(iter):
            result = self.run_single_glauber(verbose=verbose)
            results.append(result)
            fixations += result["fixation"]
            if result["fixation"]:
                iterations += result["iterations"]

        mean_iterations = iterations / fixations if fixations > 0 else 0
        return {
            "fixation_rate": fixations / iter,
            "mean_iterations_when_fix": mean_iterations,
            "p": self.p,
            "n_outer": self.n_outer,
            "n_inner": self.n_interior,
            "t": self.t,
        }
