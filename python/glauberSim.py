from abc import ABC, abstractmethod
import numpy as np

class GlauberSim(ABC):

    def __init__(
        self,
        n_outer: np.int64,
        n_interior: np.int64,
        p: np.float64,
        t: np.int64,
        tol: np.float64,
    ) -> None:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1

        Parameters
        ----------
        n_outer : int
            dimension of outer lattice
        n_inner : int
            dimension of inner lattice
        p : float
            probability that vertices are -1 at t=0
        t : int
            number of vertex-updates to perform
        tol : float
            minimum share of -1 vertices to reach to declare fixation
        """
        self.n_outer = n_outer
        self.n_interior = n_interior
        self.p = p
        self.t = t
        self.tol = tol

    @abstractmethod
    def run_single_glauber( *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def run_fixation_simulation(*args, **kwargs) -> dict:
        pass
