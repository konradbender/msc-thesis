import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
import datetime
import os
from numba import jit, njit, prange
from numba.typed import Dict
from numba.core import types

LOGGING_STEP = 1000
BOUNDARY = 1

@njit(cache = True, parallel = False)
def run_single_glauber(
    n_outer: np.int64, n_interior: np.int64, p: np.float64, t: np.int64, tol: np.float64, run_id: int = None,
    verbose: bool = False
) -> dict:
    """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
    with probability p of initializing a vertex to 1"""

    t = int(t)

    matrix = np.random.binomial(n=1, p=p, size=n_outer**2).astype(np.bool_)
    matrix = matrix.reshape((n_outer, n_outer))
    matrix[0, :] = BOUNDARY
    matrix[-1, :] = BOUNDARY

    indices = np.random.randint(1, n_outer - 1, size=2 * t).reshape((t, 2))

    buffer = (n_outer - n_interior) // np.int64(2)
    interior_indices = (slice(buffer, -buffer), slice(buffer, -buffer))

    target = matrix[interior_indices].shape[0] * matrix[interior_indices].shape[1]

    vector = np.ones(t) * np.int64(-1)

    iterations = 0
    fixation = np.bool_(False)
    for i, index in enumerate(indices):
        iterations += 1
        """Updates the vertex at index in the matrix"""

        nb_sum = (
            np.int64(matrix[index[0] + 1, index[1]])
            + np.int64(matrix[index[0], index[1] + 1])
            + np.int64(matrix[index[0] - 1, index[1]])
            + np.int64(matrix[index[0], index[1] - 1])
        )

        # d is half the number of neighbors
        if nb_sum > np.int64(2):
            # again add one to index because of buffer
            matrix[index[0], index[1]] = 1

        # d is half the number of neighbors
        if nb_sum < np.int64(2):
            # again add one to index because of buffer
            matrix[index[0], index[1]] = 0

        if nb_sum == np.int64(2):
            # flip coin
            z = np.random.binomial(n=1, p=np.float64(0.5))
            matrix[index[0], index[1]] = z

        summed_array = np.sum(matrix[interior_indices].flatten())

        vector[i] = summed_array / target

        if summed_array >= tol * target:
            fixation = np.bool_(True)
            break
        elif summed_array <= (1 - tol) * target:
            fixation = np.bool_(False)
            break

        if (iterations % LOGGING_STEP == 0) and verbose:
            print("iteration:", iterations, "share of 1 is:", (summed_array / target))
    
    result = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:]
    )
    result.update({"fixation": np.asarray([np.float64(fixation)]),
                    "iterations": np.asarray([np.float64(iterations)]), 
                    "vector": vector})

    return result

@njit(cache = True, parallel = False)
def run_fixation_simulation(
    n_outer: int, n_inner: int, p: float, t: int, iter: int, tol: float,
    verbose: bool = False
) -> dict:
    fixations = 0
    iterations = 0
    results = []

    for i in prange(iter):
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
