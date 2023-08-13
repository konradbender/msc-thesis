"""
g++ -c -std=c++11 -fPIC glauber.cpp -o glauberC.o
g++ -shared -o libGlauberC.so  glauberC.o 
"""

# write a file for each glauber run, read it in python, delete it 
# and return numpy values as normal glauber python file

from ctypes import cdll, c_double, c_int, c_char_p, create_string_buffer
import datetime
import numpy as np
import os
from glauberSim import GlauberSim

class GlauberSimCpp(GlauberSim):
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


    def run_single_glauber(self, call_id: int, verbose) -> dict:

        os.makedirs("temp", exist_ok=True)

        assert(call_id < 9999)

        now = datetime.datetime.now()
        time_string = now.strftime('%m%d%H%M%S')

        b_time_string = time_string.encode('utf-8')
        b_time_call = str(call_id).encode('utf-8')

        lib = cdll.LoadLibrary("./c++/libGlauberC.so")
        fixation = lib.run_single_c(c_int(self.n_outer), c_int(self.n_interior), c_double(self.p), c_int(self.t), 
                                    c_double(self.tol), 
                                    create_string_buffer(b_time_string),
                                    create_string_buffer(b_time_call))
        
        trace = np.loadtxt(f'./temp/{time_string}_{call_id}.txt')

        iterations = np.argmin(trace) # in case of multiple occurences, the first is returned.


        return {'fixation': fixation, 'iterations': iterations, 'vector': trace}

        pass


    def run_fixation_simulation(n_outer: int, n_inner: int, p: float, t:int, iter:int, tol: float) -> dict:
        raise NotImplementedError()
        """
        return {
            "fixation_rate": fixations / iter,
            "mean_iterations_when_fix": mean_iterations,
            "p": p,
            "n_outer": n_outer,
            "n_inner": n_inner,
            "t": t,
        }
        """

    def run_simulation(self):

        lib = cdll.LoadLibrary("./c++/libGlauberC.so")

        fixations = 0
        iterations = 0

        for i in range(iter):
            result = lib.run_single_c(self.n_outer, self.n_inner, c_double(self.p), self.t, 
                                    c_double(self.tol))

            if result:
                fixations += 1


        return {'fixation_rate': fixations / iter,
                "p": p, "n_outer": self.n_outer,"n_inner": self.n_inner, "t": self.t}




   