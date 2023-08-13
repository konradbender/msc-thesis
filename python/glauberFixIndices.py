import numpy as np
import logging
import json
import datetime
import os

from bitarray import bitarray as ba
import bitarray

from DataStructs.BitArrayMat import BitArrayMat

from glauberSim import GlauberSim


LOGGING_STEP = 1000
BOUNDARY = 1


class GlauberSimulatorFixIndices(GlauberSim):


    def run_single_glauber(
        self,
        run_id: int = None,
        verbose: bool = False
    ) -> dict:
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
        run_id : string
            identification of this glauber run to retrieve files saved to disk
        verbose : bool
            if print statements should be performed
        """
        

        matrix = np.random.binomial(n=1, p=self.p, size=self.n_outer**2).astype(np.bool_)
        matrix = matrix.reshape((self.n_outer, self.n_outer))
        matrix[0, :] = BOUNDARY
        matrix[-1, :] = BOUNDARY

        matrix = BitArrayMat(self.n_outer, self.n_outer, matrix.flatten().tolist())

        # list of indices we want to look at
        indices = np.random.randint(1, self.n_outer - 1, size=2 * self.t).reshape((self.t, 2))

        # this is the padding between inner and outer lattice
        buffer = (self.n_outer - self.n_interior) // 2

        # make a bitarray for the mask for the inner lattice
        interior_mask = np.zeros((self.n_outer, self.n_outer), dtype=np.bool_)
        interior_mask[buffer:-buffer, buffer:-buffer] = True
        interior_mask =  ba(interior_mask.flatten().tolist())

        target = self.n_interior**2

        vector = np.ones(self.t) * np.int64(-1)

        iterations = 0
        fixation = np.bool_(False)

        for i, index in enumerate(indices):
            iterations += 1
            """Updates the vertex at index in the matrix"""

            nb_sum = (
                matrix[index[0] + 1, index[1]] +
                matrix[index[0], index[1] + 1] +
                matrix[index[0] - 1, index[1]] +
                matrix[index[0], index[1] - 1] 
            )

            # d is half the number of neighbors
            if nb_sum > 2:
                # again add one to index because of buffer
                matrix[index[0], index[1]] = 1

            # d is half the number of neighbors
            if nb_sum < 2:
                # again add one to index because of buffer
                matrix[index[0], index[1]] = 0

            if nb_sum == 2:
                # flip coin
                z = np.random.binomial(n=1, p=np.float64(0.5))
                matrix[index[0], index[1]] = z

            summed_array = matrix.count(1)

            vector[i] = summed_array / target

            if summed_array >= self.tol * target:
                fixation = np.bool_(True)
                break
            elif summed_array <= (1 - self.tol) * target:
                fixation = np.bool_(False)
                break

            if (iterations % LOGGING_STEP == 0) and verbose:
                print("iteration:", iterations, "share of 1 is:", (summed_array / target))

        # end glauber for loop
        
        result =  {}

        result.update({"fixation": np.asarray([np.float64(fixation)]),
                        "iterations": np.asarray([np.float64(iterations)]), 
                        "vector": vector})

        return result

    def run_fixation_simulation(
        self,
        iter,
        verbose: bool = False
    ) -> dict:
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
