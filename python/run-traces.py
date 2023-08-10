import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
import datetime
import os
import glauber2D
import glauberFast
from copy import deepcopy
from numba import njit, prange

@njit(cache = True, parallel = False, debug=True)
def run_traces(n_outer, n_interior, t, tol, iterations, p):
    n_outer = np.int64(n_outer)
    n_interior = np.int64(n_interior)
    t = np.int64(t)
    tol = np.float64(tol)
    iterations = np.int64(iterations)
    p = np.float64(p)

    traces = np.empty(shape=(iterations, t), dtype=np.float64)
    fixations = np.empty(shape=(iterations, 1), dtype=np.float64)

    for i in prange(iterations):
        result = glauber2D.run_single_glauber(np.int64(n_outer), 
            np.int64(n_interior), 
            np.float64(p), 
            np.int64(t),
            np.float64(tol), np.int64(i), verbose = True
        )

        traces[i] = result["vector"]
        fixations[i] = result["fixation"]

    return traces, fixations


if __name__ == "__main__":
    np.random.seed(0)

    now = datetime.datetime.now()
    time_string = now.strftime("%m%d_%H%M")

    os.makedirs("./trace-results", exist_ok=True)

    logging.basicConfig(level=logging.DEBUG)

    # dimension of the lattice B'
    N_OUTER = 1000
    # dimension of the inner lattice B
    N_INTERIOR = 990

    # number of time steps for each glauber dynamics iteration
    # T = int(20_000_000)
    T = 1_000

    # number of times we run glauber dynamcis for each probability
    ITERATIONS = 4

    # tolerance for fixation
    TOL = 0.85

    P = 0.505

    traces, fixations = run_traces(N_OUTER, N_INTERIOR, T, TOL, ITERATIONS, P)

    results = {
        "n_outer": N_OUTER,
        "n_inner": N_INTERIOR,
        "t": T,
        "tol": TOL,
        "iterations": ITERATIONS,
        "time_string": time_string,
        "traces": traces,
        "fixations": fixations,
        "p": P,
    }

    results["traces"] = results["traces"].tolist()
    results["fixations"] = results["fixations"].tolist()    


    json.dump(results, open(f"./trace-results/results-1M-{time_string}.json", "w"))

    print("finished", time_string, ", took ", datetime.datetime.now() - now, " to run")
