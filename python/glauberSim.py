from abc import ABC, abstractmethod
import numpy as np

class GlauberSim(ABC):

    def __init__(
        self,
        n_interior: np.int64,
        p: np.float64,
        t: np.int64,
        tol: np.float64,
        padding : np.int64 = None, # for new code
        n_outer: np.int64 = None # kept for depreceated compatibility
    ) -> None:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1

        Parameters
        ----------
        padding : int
            number of vertices beyond the border of n_interior*n_interior matrix
        n_interior : int
            dimension of inner lattice
        p : float
            probability that vertices are -1 at t=0
        t : int
            number of vertex-updates to perform
        tol : float
            minimum share of -1 vertices to reach to declare fixation
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

        

    @abstractmethod
    def run_single_glauber( *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def run_fixation_simulation(*args, **kwargs) -> dict:
        pass
