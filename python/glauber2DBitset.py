import numpy as np
import logging
import json
import datetime
import os

from bitarray import bitarray as ba
import bitarray


LOGGING_STEP = 1000
BOUNDARY = 1

class BitArrayMat():

    def __init__(self, nrow, ncol, list) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.size = nrow*ncol
        assert(nrow*ncol == len(list))
        self.arr = ba(list)

    def idx(self, r, c):
        assert 0 <= r < self.nrow
        assert 0 <= c < self.ncol
        return r * self.ncol + c


    def __getitem__(self, key):
        if type(key) == bitarray.bitarray:
            return self.arr[key]
        else:
            x, y = key
            flat_idx = self.idx(x,y)
            return self.arr[flat_idx]
        
        
    def __setitem__(self, key, value):
        if type(key) == bitarray.bitarray:
            raise NotImplementedError
        else:
            x, y = key
            flat_idx = self.idx(x,y)
            self.arr[flat_idx] = value
        
    def count(self, i):
        assert (i==0 or i==1)
        return self.arr.count(i)
       


def run_single_glauber(
    n_outer: np.int64, n_interior: np.int64, p: np.float64, t: np.int64, tol: np.float64, run_id: int = None,
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
    

    matrix = np.random.binomial(n=1, p=p, size=n_outer**2).astype(np.bool_)
    matrix = matrix.reshape((n_outer, n_outer))
    matrix[0, :] = BOUNDARY
    matrix[-1, :] = BOUNDARY

    matrix = BitArrayMat(n_outer, n_outer, matrix.flatten().tolist())

    # list of indices we want to look at
    indices = np.random.randint(1, n_outer - 1, size=2 * t).reshape((t, 2))

    # this is the padding between inner and outer lattice
    buffer = (n_outer - n_interior) // 2

    # make a bitarray for the mask for the inner lattice
    interior_mask = np.zeros((n_outer, n_outer), dtype=np.bool_)
    interior_mask[buffer:-buffer, buffer:-buffer] = True
    interior_mask =  ba(interior_mask.flatten().tolist())

    target = n_interior**2

    vector = np.ones(t) * np.int64(-1)

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

        if summed_array >= tol * target:
            fixation = np.bool_(True)
            break
        elif summed_array <= (1 - tol) * target:
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
    n_outer: int, n_inner: int, p: float, t: int, iter: int, tol: float,
    verbose: bool = False
) -> dict:
    fixations = 0
    iterations = 0
    results = []

    for i in range(iter):
        result = run_single_glauber(n_outer, n_inner, p, t, tol, verbose=verbose)
        results.append(result)
        fixations += result["fixation"]
        if result["fixation"]:
            iterations += result["iterations"]

    mean_iterations = iterations / fixations if fixations > 0 else 0
    return {
        "fixation_rate": fixations / iter,
        "mean_iterations_when_fix": mean_iterations,
        "p": p,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "t": t,
    }
